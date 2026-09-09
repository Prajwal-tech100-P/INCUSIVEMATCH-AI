// Main JavaScript file for Inclusive Match AI

document.addEventListener('DOMContentLoaded', () => {
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    [...tooltipTriggerList].forEach(el => new bootstrap.Tooltip(el));

    if (typeof io !== 'undefined' && document.body.dataset.authenticated === 'true') {
        const socket = io({ transports: ['websocket', 'polling'] });
        window.inclusiveSocket = socket;

        socket.on('connect', () => {
            socket.emit('join_user_room', {});
            checkPendingIncomingCall();
        });

        socket.on('connect_error', (error) => {
            console.error('Socket.IO connection failed:', error);
            // HTTP polling below provides a fallback for incoming-call alerts.
        });

        socket.on('match_notification', (data) => {
            showRealtimeNotification(data);
        });

        socket.on('new_message', (data) => {
            if (data && data.text) showRealtimeNotification({
                type: 'message',
                title: `New message from ${data.sender_name || 'your match'}`,
                message: data.text,
                partner_id: data.sender_id
            });
        });

        // Incoming video/audio call invitation.
        socket.on('incoming_call', (data) => {
            if (!data || !data.caller_id || String(data.caller_id) === String(getCurrentUserId())) return;
            showIncomingCall(data, socket);
        });

        // These events are mainly useful while the caller is already on /video/<id>.
        socket.on('call_declined', (data) => {
            showRealtimeNotification({
                type: 'call',
                title: 'Call declined',
                message: `${data && data.user_name ? data.user_name : 'The user'} declined your call.`
            });
        });
    }

    // Call buttons can appear on Discover or Chat pages.
    document.querySelectorAll('.js-start-call').forEach(button => {
        button.addEventListener('click', () => {
            const partnerId = button.dataset.partnerId;
            const partnerName = button.dataset.partnerName || 'your match';
            startOutgoingCall(partnerId, partnerName);
        });
    });

    console.log('Inclusive Match AI initialized.');
});

function getCurrentUserId() {
    return document.body.dataset.userId || '';
}

function startOutgoingCall(partnerId, partnerName) {
    if (!partnerId || !window.inclusiveSocket) {
        showRealtimeNotification({
            title: 'Call unavailable',
            message: 'Real-time calling is not available. Please reload the page and try again.'
        });
        return;
    }

    showRealtimeNotification({
        title: 'Calling…',
        message: `Calling ${partnerName}.`
    });

    // Wait for the server acknowledgement before navigating so the invitation
    // is not lost during page navigation.
    window.inclusiveSocket.emit(
        'call_user',
        { partner_id: partnerId },
        (ack) => {
            if (ack && ack.ok === false) {
                showRealtimeNotification({
                    title: 'Call unavailable',
                    message: ack.message || 'You cannot call this user.'
                });
                return;
            }
            window.location.href = `/video/${encodeURIComponent(partnerId)}?outgoing=1`;
        }
    );
}

function showIncomingCall(data, socket) {
    // Only show one incoming-call dialog at a time.
    const old = document.getElementById('incoming-call-overlay');
    if (old) old.remove();

    const callerName = escapeHtml(data.caller_name || 'Someone');
    const callerId = escapeHtml(data.caller_id);

    const overlay = document.createElement('div');
    overlay.id = 'incoming-call-overlay';
    overlay.innerHTML = `
        <div class="incoming-call-backdrop"></div>
        <section class="incoming-call-card" role="dialog" aria-modal="true"
                 aria-labelledby="incoming-call-title">
            <div class="incoming-call-icon" aria-hidden="true">
                <i class="bi bi-camera-video-fill"></i>
            </div>
            <div class="incoming-call-label">INCOMING CALL</div>
            <h2 id="incoming-call-title">${callerName}</h2>
            <p>${callerName} is calling you</p>
            <div class="incoming-call-actions">
                <button type="button" class="incoming-decline" id="incoming-decline">
                    <i class="bi bi-telephone-x-fill"></i>
                    <span>Decline</span>
                </button>
                <button type="button" class="incoming-accept" id="incoming-accept">
                    <i class="bi bi-camera-video-fill"></i>
                    <span>Accept</span>
                </button>
            </div>
        </section>
    `;

    document.body.appendChild(overlay);

    const cleanup = () => {
        overlay.remove();
    };

    document.getElementById('incoming-decline').onclick = () => {
        socket.emit('decline_call', { caller_id: data.caller_id, call_id: data.call_id });
        cleanup();
        showRealtimeNotification({
            title: 'Call declined',
            message: `You declined ${data.caller_name || 'the incoming call'}.`
        });
    };

    document.getElementById('incoming-accept').onclick = () => {
        socket.emit('accept_call', { caller_id: data.caller_id, call_id: data.call_id }, (ack) => {
            if (ack && ack.ok === false) {
                cleanup();
                showRealtimeNotification({
                    title: 'Call unavailable',
                    message: ack.message || 'This call is no longer available.'
                });
                return;
            }

            cleanup();
            window.location.href = `/video/${encodeURIComponent(data.caller_id)}?incoming=1`;
        });
    };
}

function showRealtimeNotification(data) {
    const title = data.title || 'Notification';
    const message = data.message || '';
    const toast = document.createElement('div');
    toast.className = 'position-fixed top-0 end-0 m-3 alert alert-light shadow border-0 rounded-4';
    toast.style.zIndex = '2000';
    toast.style.maxWidth = '380px';
    toast.innerHTML = `<div class="fw-bold">${escapeHtml(title)}</div><div class="small text-muted mt-1">${escapeHtml(message)}</div>`;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 6000);
}

let pendingCallRequest = false;

async function checkPendingIncomingCall() {
    if (pendingCallRequest || document.hidden) return;
    pendingCallRequest = true;
    try {
        const response = await fetch('/api/calls/pending', {
            method: 'GET',
            credentials: 'same-origin',
            cache: 'no-store',
            headers: {'Accept': 'application/json'}
        });
        if (!response.ok) return;
        const data = await response.json();
        if (!data.pending || !data.caller_id) return;
        if (String(data.caller_id) === String(getCurrentUserId())) return;
        if (!document.getElementById('incoming-call-overlay') && window.inclusiveSocket) {
            showIncomingCall(data, window.inclusiveSocket);
        }
    } catch (error) {
        console.warn('Pending call check failed:', error);
    } finally {
        pendingCallRequest = false;
    }
}

setInterval(checkPendingIncomingCall, 2000);
window.addEventListener('focus', checkPendingIncomingCall);
document.addEventListener('visibilitychange', () => {
    if (!document.hidden) checkPendingIncomingCall();
});

function escapeHtml(value) {
    return String(value).replace(/[&<>'"]/g, ch => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
    }[ch]));
}
