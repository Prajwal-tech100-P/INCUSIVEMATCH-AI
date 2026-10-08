// Main JavaScript file for Inclusive Match AI

document.addEventListener('DOMContentLoaded', () => {
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    [...tooltipTriggerList].forEach(el => new bootstrap.Tooltip(el));

    if (typeof Notification !== 'undefined' && Notification.permission === 'default') {
        document.addEventListener('click', function reqNotif() {
            Notification.requestPermission();
            document.removeEventListener('click', reqNotif);
        });
    }

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

        // These events are mainly useful while the caller is already on /video/<id>, or waiting on the caller screen.
        socket.on('call_declined', (data) => {
            if ((currentOutgoingCallId && String(data.call_id) === String(currentOutgoingCallId)) || 
                (currentOutgoingPartnerId && String(data.user_id) === String(currentOutgoingPartnerId))) {
                cleanupOutgoingCallUI();
            }
            showRealtimeNotification({
                type: 'call',
                title: 'Call declined',
                message: `${data && data.user_name ? data.user_name : 'The user'} declined your call.`
            });
        });

        socket.on('call_cancelled', (data) => {
            const notif = document.getElementById(`call-notification-${data.call_id}`);
            if (notif) {
                const txt = notif.querySelector('.call-notif-msg');
                if (txt) txt.textContent = "This call has ended.";
                const actions = document.getElementById(`call-actions-${data.call_id}`);
                if (actions) actions.remove();
                notif.querySelector('.dropdown-item').classList.remove('border-primary', 'bg-light');
                decrementNotificationBadge();
            }
            const old = document.getElementById('incoming-call-overlay');
            if (old) old.remove();
            stopRingtone();
        });

        socket.on('call_accepted', (data) => {
            if ((currentOutgoingCallId && String(data.call_id) === String(currentOutgoingCallId)) ||
                (currentOutgoingPartnerId && String(data.user_id) === String(currentOutgoingPartnerId))) {
                cleanupOutgoingCallUI();
                window.location.href = `/video/${encodeURIComponent(data.user_id)}?outgoing=1`;
            }
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

let incomingRingtone = null;

function playRingtone() {
    if (!incomingRingtone) {
        incomingRingtone = new Audio('/static/audio/incoming_call.mp3');
        incomingRingtone.loop = true;
    }
    // Handle autoplay restrictions gracefully
    incomingRingtone.play().catch(e => console.warn('Ringtone autoplay blocked:', e));
}

function stopRingtone() {
    if (incomingRingtone) {
        incomingRingtone.pause();
        incomingRingtone.currentTime = 0;
    }
}

function getCurrentUserId() {
    return document.body.dataset.userId || '';
}

let currentOutgoingCallId = null;
let currentOutgoingPartnerId = null;

function showOutgoingCallUI(partnerId, partnerName) {
    const old = document.getElementById('outgoing-call-overlay');
    if (old) old.remove();

    const overlay = document.createElement('div');
    overlay.id = 'outgoing-call-overlay';
    overlay.innerHTML = `
        <div style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: linear-gradient(135deg, #4f46e5, #6366f1); z-index: 9999; display: flex; flex-direction: column; align-items: center; justify-content: center; color: white; font-family: inherit;">
            <h2 style="font-weight: 700; margin-bottom: 2rem;">Inclusive Match AI</h2>
            <img src="https://ui-avatars.com/api/?name=${encodeURIComponent(partnerName)}&background=fff&color=6366f1&size=100" style="border-radius: 50%; width: 100px; height: 100px; margin-bottom: 1rem; object-fit: cover; box-shadow: 0 8px 16px rgba(0,0,0,0.2);" alt="Avatar">
            <h3 style="font-weight: 600; margin-bottom: 1.5rem;">${escapeHtml(partnerName)}</h3>
            <p style="font-size: 1.25rem; font-weight: 500; margin-bottom: 0.5rem;">Calling...</p>
            <p style="font-size: 1rem; opacity: 0.9; margin-bottom: 0.5rem;">Calling ${escapeHtml(partnerName)}</p>
            <p style="font-size: 0.9rem; opacity: 0.75; margin-bottom: 3rem;">Waiting for ${escapeHtml(partnerName)} to accept...</p>
            <button id="outgoing-cancel-btn" style="background: #ef4444; color: white; border: none; padding: 12px 32px; border-radius: 50px; font-size: 1.1rem; font-weight: 600; cursor: pointer; transition: background 0.2s; box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4);">
                <i class="bi bi-telephone-x-fill" style="margin-right: 8px;"></i> End Call
            </button>
        </div>
    `;
    document.body.appendChild(overlay);

    document.getElementById('outgoing-cancel-btn').onclick = () => {
        if (window.inclusiveSocket && currentOutgoingCallId && currentOutgoingPartnerId) {
            window.inclusiveSocket.emit('cancel_call', {
                call_id: currentOutgoingCallId,
                partner_id: currentOutgoingPartnerId
            });
        }
        cleanupOutgoingCallUI();
    };
}

function cleanupOutgoingCallUI() {
    const overlay = document.getElementById('outgoing-call-overlay');
    if (overlay) overlay.remove();
    currentOutgoingCallId = null;
    currentOutgoingPartnerId = null;
}

function startOutgoingCall(partnerId, partnerName) {
    if (!partnerId || !window.inclusiveSocket) {
        showRealtimeNotification({
            title: 'Call unavailable',
            message: 'Real-time calling is not available. Please reload the page and try again.'
        });
        return;
    }

    showOutgoingCallUI(partnerId, partnerName);

    window.inclusiveSocket.emit(
        'call_user',
        { partner_id: partnerId },
        (ack) => {
            if (ack && ack.ok === false) {
                cleanupOutgoingCallUI();
                showRealtimeNotification({
                    title: 'Call unavailable',
                    message: ack.message || 'You cannot call this user.'
                });
                return;
            }
            currentOutgoingCallId = ack.call_id;
            currentOutgoingPartnerId = partnerId;
        }
    );
}

function incrementNotificationBadge() {
    const badge = document.getElementById('notificationBadge');
    if (badge) {
        let count = parseInt(badge.textContent) || 0;
        badge.textContent = count + 1;
        badge.classList.remove('d-none');
    }
}

function decrementNotificationBadge() {
    const badge = document.getElementById('notificationBadge');
    if (badge) {
        let count = parseInt(badge.textContent) || 0;
        count = Math.max(0, count - 1);
        badge.textContent = count;
        if (count === 0) badge.classList.add('d-none');
    }
}

function showIncomingCall(data, socket) {
    const callId = escapeHtml(data.call_id);
    const existing = document.getElementById(`call-notification-${callId}`);
    if (existing) return; // Already showing

    const oldOverlay = document.getElementById('incoming-call-overlay');
    if (oldOverlay) oldOverlay.remove(); // Just in case

    const callerName = escapeHtml(data.caller_name || 'Someone');
    const callerId = escapeHtml(data.caller_id);

    const li = document.createElement('li');
    li.id = `call-notification-${callId}`;
    li.innerHTML = `
        <div class="dropdown-item rounded-3 py-2 bg-light border border-primary border-2" style="cursor: default;">
            <div class="d-flex align-items-center mb-2">
                <i class="bi bi-camera-video-fill text-primary fs-4 me-2"></i>
                <div>
                    <span class="fw-bold d-block text-primary">Incoming Video Call</span>
                    <span class="small text-dark call-notif-msg">${callerName} is calling you.</span>
                </div>
            </div>
            <div class="d-flex gap-2 mt-2" id="call-actions-${callId}">
                <button class="btn btn-success btn-sm flex-grow-1 incoming-accept">
                    <i class="bi bi-telephone-inbound-fill me-1"></i> Answer
                </button>
                <button class="btn btn-danger btn-sm flex-grow-1 incoming-decline">
                    <i class="bi bi-telephone-x-fill me-1"></i> Decline
                </button>
            </div>
        </div>
    `;

    const list = document.getElementById('notificationList');
    if (list) {
        list.insertBefore(li, list.children[1]); // Insert after header
        incrementNotificationBadge();
        showRealtimeNotification({
            title: 'Incoming Video Call',
            message: `${callerName} is calling you. Check your notifications 🔔.`
        });
    }

    playRingtone();

    const acceptBtn = li.querySelector('.incoming-accept');
    const declineBtn = li.querySelector('.incoming-decline');

    const cleanup = () => {
        stopRingtone();
        const actions = document.getElementById(`call-actions-${callId}`);
        if (actions) {
            actions.remove();
            decrementNotificationBadge();
        }
    };

    declineBtn.onclick = (e) => {
        e.stopPropagation();
        socket.emit('decline_call', { caller_id: data.caller_id, call_id: data.call_id });
        cleanup();
        showRealtimeNotification({
            title: 'Call declined',
            message: `You declined ${data.caller_name || 'the incoming call'}.`
        });
        li.remove();
    };

    acceptBtn.onclick = (e) => {
        e.stopPropagation();
        socket.emit('accept_call', { caller_id: data.caller_id, call_id: data.call_id }, (ack) => {
            if (ack && ack.ok === false) {
                cleanup();
                const txt = li.querySelector('.call-notif-msg');
                if (txt) txt.textContent = "This call is no longer available.";
                showRealtimeNotification({
                    title: 'Call unavailable',
                    message: ack.message || 'This call is no longer available.'
                });
                return;
            }
            cleanup();
            li.remove();
            window.location.href = `/video/${encodeURIComponent(data.caller_id)}?incoming=1`;
        });
    };
}

function showRealtimeNotification(data) {
    const title = data.title || 'Notification';
    const message = data.message || '';
    
    if (document.hidden && typeof Notification !== 'undefined') {
        if (Notification.permission === 'granted') {
            const notif = new Notification(title, { body: message });
            notif.onclick = () => { window.focus(); notif.close(); };
            
            // Try to play a quick notification sound if available
            try {
                const audio = new Audio('/static/audio/incoming_call.mp3');
                audio.play();
                setTimeout(() => audio.pause(), 2000); // Play only 2 seconds for a short ping
            } catch(e) {}
        }
    }

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
        
        const existingOverlay = document.getElementById('incoming-call-overlay');
        
        if (!data.pending || !data.caller_id) {
            // Call has expired or been cancelled
            const existingNotifs = document.querySelectorAll('[id^="call-notification-"]');
            existingNotifs.forEach(n => {
                const actions = n.querySelector('[id^="call-actions-"]');
                if (actions) {
                    actions.remove();
                    decrementNotificationBadge();
                    const txt = n.querySelector('.call-notif-msg');
                    if (txt) txt.textContent = "This call has ended.";
                    n.querySelector('.dropdown-item').classList.remove('border-primary', 'bg-light');
                }
            });
            const existingOverlay = document.getElementById('incoming-call-overlay');
            if (existingOverlay) {
                existingOverlay.remove();
            }
            stopRingtone();
            return;
        }
        
        if (String(data.caller_id) === String(getCurrentUserId())) return;
        if (!existingOverlay && window.inclusiveSocket) {
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
