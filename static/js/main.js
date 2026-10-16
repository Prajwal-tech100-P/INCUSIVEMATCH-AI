// Main JavaScript file for Inclusive Match AI

let incomingRingtone = null;
let isRingtonePlaying = false;
let syntheticRingtoneTimer = null;
let ringtoneAudioContext = null;
let messageSoundAudio = null;
const seenMessageIds = new Set();

function getCurrentUserId() {
    return document.body.dataset.userId || '';
}

function isMessageSoundEnabled() {
    return localStorage.getItem('inclusive_msg_sound_enabled') !== 'false';
}

function playMessageNotificationSound() {
    if (!isMessageSoundEnabled()) return;

    try {
        if (!messageSoundAudio) {
            messageSoundAudio = new Audio('/static/audio/message_notification.mp3');
            messageSoundAudio.preload = 'auto';
            messageSoundAudio.onerror = () => {
                console.warn('Could not load message_notification.mp3, using synthetic chime fallback.');
                playSyntheticChime();
            };
        }
        messageSoundAudio.currentTime = 0;
        const p = messageSoundAudio.play();
        if (p && typeof p.catch === 'function') {
            p.catch(err => {
                console.warn('Message notification sound blocked or failed:', err);
                playSyntheticChime();
            });
        }
    } catch (e) {
        console.warn('Message sound audio error:', e);
        playSyntheticChime();
    }
}

function playSyntheticChime() {
    if (!isMessageSoundEnabled()) return;
    try {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (!AudioCtx) return;
        const ctx = new AudioCtx();
        const now = ctx.currentTime;

        // Tone 1: 587.33 Hz (D5)
        const osc1 = ctx.createOscillator();
        const gain1 = ctx.createGain();
        osc1.type = 'sine';
        osc1.frequency.setValueAtTime(587.33, now);
        gain1.gain.setValueAtTime(0.12, now);
        gain1.gain.exponentialRampToValueAtTime(0.001, now + 0.15);
        osc1.connect(gain1);
        gain1.connect(ctx.destination);
        osc1.start(now);
        osc1.stop(now + 0.15);

        // Tone 2: 880 Hz (A5)
        const osc2 = ctx.createOscillator();
        const gain2 = ctx.createGain();
        osc2.type = 'sine';
        osc2.frequency.setValueAtTime(880, now + 0.1);
        gain2.gain.setValueAtTime(0.12, now + 0.1);
        gain2.gain.exponentialRampToValueAtTime(0.001, now + 0.3);
        osc2.connect(gain2);
        gain2.connect(ctx.destination);
        osc2.start(now + 0.1);
        osc2.stop(now + 0.3);
    } catch (e) {
        // Web Audio fallback blocked or not supported
    }
}

function playRingtone() {
    if (isRingtonePlaying) return; // Prevent multiple simultaneous ringtones
    isRingtonePlaying = true;

    try {
        if (!incomingRingtone) {
            incomingRingtone = new Audio('/static/audio/incoming_call.mp3');
            incomingRingtone.loop = true;
            incomingRingtone.preload = 'auto';
            incomingRingtone.onerror = () => {
                console.warn('incoming_call.mp3 could not be loaded, using synthetic phone ringtone.');
                startSyntheticRingtone();
            };
        }
        incomingRingtone.currentTime = 0;
        const playPromise = incomingRingtone.play();
        if (playPromise && typeof playPromise.catch === 'function') {
            playPromise.catch(error => {
                console.warn('Ringtone autoplay blocked by browser policy:', error);
                const unlockEl = document.getElementById('incoming-audio-unlock');
                if (unlockEl) {
                    unlockEl.classList.remove('d-none');
                }
                startSyntheticRingtone();
            });
        }
    } catch (e) {
        console.warn('Error starting ringtone:', e);
        startSyntheticRingtone();
    }
}

function unlockRingtoneAudio() {
    if (incomingRingtone && isRingtonePlaying) {
        incomingRingtone.play().then(() => {
            const unlockEl = document.getElementById('incoming-audio-unlock');
            if (unlockEl) unlockEl.classList.add('d-none');
            stopSyntheticRingtone();
        }).catch(e => console.warn('Unlock audio failed:', e));
    }
}

function startSyntheticRingtone() {
    if (syntheticRingtoneTimer) return;
    const playRingBurst = () => {
        if (!isRingtonePlaying) return;
        try {
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (!AudioCtx) return;
            if (!ringtoneAudioContext || ringtoneAudioContext.state === 'closed') {
                ringtoneAudioContext = new AudioCtx();
            }
            if (ringtoneAudioContext.state === 'suspended') {
                ringtoneAudioContext.resume().catch(() => {});
            }
            const now = ringtoneAudioContext.currentTime;

            const oscA = ringtoneAudioContext.createOscillator();
            const oscB = ringtoneAudioContext.createOscillator();
            const gainNode = ringtoneAudioContext.createGain();

            oscA.frequency.setValueAtTime(440, now);
            oscB.frequency.setValueAtTime(480, now);

            gainNode.gain.setValueAtTime(0.15, now);
            gainNode.gain.setValueAtTime(0.15, now + 1.2);
            gainNode.gain.exponentialRampToValueAtTime(0.001, now + 1.3);

            oscA.connect(gainNode);
            oscB.connect(gainNode);
            gainNode.connect(ringtoneAudioContext.destination);

            oscA.start(now);
            oscB.start(now);
            oscA.stop(now + 1.3);
            oscB.stop(now + 1.3);
        } catch (err) {
            // Web Audio blocked
        }
    };

    playRingBurst();
    syntheticRingtoneTimer = setInterval(playRingBurst, 3500);
}

function stopSyntheticRingtone() {
    if (syntheticRingtoneTimer) {
        clearInterval(syntheticRingtoneTimer);
        syntheticRingtoneTimer = null;
    }
    if (ringtoneAudioContext && ringtoneAudioContext.state !== 'closed') {
        try {
            ringtoneAudioContext.close().catch(() => {});
        } catch (e) {}
        ringtoneAudioContext = null;
    }
}

function stopRingtone() {
    isRingtonePlaying = false;
    if (incomingRingtone) {
        incomingRingtone.pause();
        incomingRingtone.currentTime = 0;
    }
    stopSyntheticRingtone();
    const unlockEl = document.getElementById('incoming-audio-unlock');
    if (unlockEl) unlockEl.classList.add('d-none');
}

// User-interaction audio unlock listener
document.addEventListener('click', () => {
    if (isRingtonePlaying && incomingRingtone && incomingRingtone.paused) {
        unlockRingtoneAudio();
    }
}, { passive: true });

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
        });

        socket.on('match_notification', (data) => {
            showRealtimeNotification(data);
        });

        // Incoming message event
        socket.on('new_message', (data) => {
            if (!data || !data.sender_id) return;
            // Never play or notify sender about their own message
            if (String(data.sender_id) === String(getCurrentUserId())) return;

            // Deduplication to prevent multiple sounds/toasts on reconnects
            const msgId = data.message_id || (data.sender_id + '_' + (data.text || '').slice(0, 30));
            if (seenMessageIds.has(msgId)) return;
            seenMessageIds.add(msgId);
            if (seenMessageIds.size > 200) {
                const first = seenMessageIds.values().next().value;
                seenMessageIds.delete(first);
            }

            // Play message sound (dedicated message chime, never incoming-call ringtone)
            playMessageNotificationSound();

            // Check if user is actively viewing this conversation
            const isActivelyViewing = (window.ACTIVE_CHAT_PARTNER_ID && String(window.ACTIVE_CHAT_PARTNER_ID) === String(data.sender_id) && !document.hidden);
            if (!isActivelyViewing && data.text) {
                showRealtimeNotification({
                    type: 'message',
                    title: `New message from ${data.sender_name || 'your match'}`,
                    message: data.text,
                    partner_id: data.sender_id
                });
            }
        });

        // Incoming video/audio call invitation
        socket.on('incoming_call', (data) => {
            if (!data || !data.caller_id || String(data.caller_id) === String(getCurrentUserId())) return;
            showIncomingCall(data, socket);
        });

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
                if (txt) txt.textContent = "This call was cancelled.";
                const actions = document.getElementById(`call-actions-${data.call_id}`);
                if (actions) actions.remove();
                notif.querySelector('.dropdown-item').classList.remove('border-primary', 'bg-light');
                decrementNotificationBadge();
            }
            const old = document.getElementById('incoming-call-overlay');
            if (old) old.remove();
            stopRingtone();

            showRealtimeNotification({
                type: 'call',
                title: 'Missed Call',
                message: 'A video call was cancelled by the caller.'
            });
        });

        socket.on('call_accepted', (data) => {
            if ((currentOutgoingCallId && String(data.call_id) === String(currentOutgoingCallId)) ||
                (currentOutgoingPartnerId && String(data.user_id) === String(currentOutgoingPartnerId))) {
                cleanupOutgoingCallUI();
                window.location.href = `/video/${encodeURIComponent(data.user_id)}?outgoing=1`;
            }
        });
    }

    // Call buttons on Discover or Chat pages
    document.querySelectorAll('.js-start-call').forEach(button => {
        button.addEventListener('click', () => {
            const partnerId = button.dataset.partnerId;
            const partnerName = button.dataset.partnerName || 'your match';
            startOutgoingCall(partnerId, partnerName);
        });
    });

    console.log('Inclusive Match AI initialized.');
});

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
    if (!data || !data.caller_id || String(data.caller_id) === String(getCurrentUserId())) return;

    const callId = escapeHtml(data.call_id);
    const callerName = escapeHtml(data.caller_name || 'Someone');
    const callerId = escapeHtml(data.caller_id);
    const callerAvatar = escapeHtml(data.caller_avatar || `https://ui-avatars.com/api/?name=${encodeURIComponent(data.caller_name || 'User')}&background=6366f1&color=fff&size=120`);

    // Prevent duplicate overlay
    const existingOverlay = document.getElementById('incoming-call-overlay');
    if (existingOverlay) {
        if (existingOverlay.dataset.callId === String(data.call_id)) return;
        existingOverlay.remove();
    }

    // Play incoming ringtone loop
    playRingtone();

    // 1. Build and show full modal overlay
    const overlay = document.createElement('div');
    overlay.id = 'incoming-call-overlay';
    overlay.dataset.callId = String(data.call_id);
    overlay.style.cssText = 'position:fixed; top:0; left:0; width:100vw; height:100vh; background:rgba(15,23,42,0.85); backdrop-filter:blur(8px); z-index:99999; display:flex; align-items:center; justify-content:center; animation:fadeIn 0.2s ease;';

    overlay.innerHTML = `
        <div class="incoming-call-card text-center p-4 p-md-5 rounded-4 shadow-lg text-white" style="background: linear-gradient(145deg, #1e293b, #0f172a); border: 1px solid rgba(255,255,255,0.18); max-width: 420px; width: 92%;">
            <div class="mb-3 position-relative d-inline-block">
                <img src="${callerAvatar}" alt="${callerName}" class="rounded-circle shadow" style="width: 110px; height: 110px; object-fit: cover; border: 4px solid #6366f1;" onerror="this.src='https://ui-avatars.com/api/?name=${encodeURIComponent(data.caller_name || 'User')}&background=6366f1&color=fff&size=110'">
            </div>
            <h3 class="fw-bold mb-1 text-white">${callerName}</h3>
            <p class="text-info-emphasis mb-3 small"><i class="bi bi-camera-video me-1"></i> Incoming Video Call…</p>
            
            <!-- Visible Audio Unlock helper if autoplay is blocked -->
            <div id="incoming-audio-unlock" class="alert alert-warning py-2 px-3 small rounded-pill d-none mb-3 text-start" style="font-size:0.8rem; cursor:pointer;" onclick="unlockRingtoneAudio()">
                <i class="bi bi-volume-mute-fill me-1"></i> Ringtone blocked. <strong class="text-decoration-underline">Click here to enable sound</strong>
            </div>

            <div class="d-flex justify-content-center gap-3 mt-3">
                <button id="incoming-modal-decline" class="btn btn-danger rounded-pill px-4 py-2 d-flex align-items-center gap-2 fw-semibold shadow">
                    <i class="bi bi-telephone-x-fill fs-5"></i> Decline
                </button>
                <button id="incoming-modal-accept" class="btn btn-success rounded-pill px-4 py-2 d-flex align-items-center gap-2 fw-semibold shadow">
                    <i class="bi bi-telephone-inbound-fill fs-5"></i> Accept
                </button>
            </div>
        </div>
    `;
    document.body.appendChild(overlay);

    // 2. Also register in navbar notification dropdown for status & badge
    const existingNotif = document.getElementById(`call-notification-${callId}`);
    if (!existingNotif) {
        const li = document.createElement('li');
        li.id = `call-notification-${callId}`;
        li.innerHTML = `
            <div class="dropdown-item rounded-3 py-2 bg-light border border-primary border-2" style="cursor: default;">
                <div class="d-flex align-items-center mb-2">
                    <img src="${callerAvatar}" class="rounded-circle me-2" width="36" height="36" style="object-fit:cover;" onerror="this.src='https://ui-avatars.com/api/?name=${encodeURIComponent(data.caller_name || 'User')}&background=6366f1&color=fff&size=36'">
                    <div>
                        <span class="fw-bold d-block text-primary">Incoming Video Call</span>
                        <span class="small text-dark call-notif-msg">${callerName} is calling you.</span>
                    </div>
                </div>
                <div class="d-flex gap-2 mt-2" id="call-actions-${callId}">
                    <button class="btn btn-success btn-sm flex-grow-1 incoming-accept">
                        <i class="bi bi-telephone-inbound-fill me-1"></i> Accept
                    </button>
                    <button class="btn btn-danger btn-sm flex-grow-1 incoming-decline">
                        <i class="bi bi-telephone-x-fill me-1"></i> Decline
                    </button>
                </div>
            </div>
        `;
        const list = document.getElementById('notificationList');
        if (list) {
            list.insertBefore(li, list.children[1]);
            incrementNotificationBadge();
        }
    }

    const cleanup = () => {
        stopRingtone();
        const o = document.getElementById('incoming-call-overlay');
        if (o) o.remove();
        const actions = document.getElementById(`call-actions-${callId}`);
        if (actions) {
            actions.remove();
            decrementNotificationBadge();
        }
    };

    const handleDecline = (e) => {
        if (e) e.stopPropagation();
        socket.emit('decline_call', { caller_id: data.caller_id, call_id: data.call_id });
        cleanup();
        const notifItem = document.getElementById(`call-notification-${callId}`);
        if (notifItem) notifItem.remove();
        showRealtimeNotification({
            title: 'Call declined',
            message: `You declined ${data.caller_name || 'the incoming call'}.`
        });
    };

    const handleAccept = (e) => {
        if (e) e.stopPropagation();
        socket.emit('accept_call', { caller_id: data.caller_id, call_id: data.call_id }, (ack) => {
            if (ack && ack.ok === false) {
                cleanup();
                const notifItem = document.getElementById(`call-notification-${callId}`);
                if (notifItem) {
                    const txt = notifItem.querySelector('.call-notif-msg');
                    if (txt) txt.textContent = "This call is no longer available.";
                }
                showRealtimeNotification({
                    title: 'Call unavailable',
                    message: ack.message || 'This call is no longer available.'
                });
                return;
            }
            cleanup();
            const notifItem = document.getElementById(`call-notification-${callId}`);
            if (notifItem) notifItem.remove();
            // Accept navigates to video call room, establishing the existing WebRTC connection
            window.location.href = `/video/${encodeURIComponent(data.caller_id)}?incoming=1`;
        });
    };

    // Modal buttons
    const modalDecline = document.getElementById('incoming-modal-decline');
    const modalAccept = document.getElementById('incoming-modal-accept');
    if (modalDecline) modalDecline.onclick = handleDecline;
    if (modalAccept) modalAccept.onclick = handleAccept;

    // Dropdown buttons
    const notifLi = document.getElementById(`call-notification-${callId}`);
    if (notifLi) {
        const acceptBtn = notifLi.querySelector('.incoming-accept');
        const declineBtn = notifLi.querySelector('.incoming-decline');
        if (acceptBtn) acceptBtn.onclick = handleAccept;
        if (declineBtn) declineBtn.onclick = handleDecline;
    }
}

function showRealtimeNotification(data) {
    const title = data.title || 'Notification';
    const message = data.message || '';

    if (document.hidden && typeof Notification !== 'undefined') {
        if (Notification.permission === 'granted') {
            const notif = new Notification(title, { body: message });
            notif.onclick = () => { window.focus(); notif.close(); };
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
            if (existingOverlay) {
                existingOverlay.remove();
                showRealtimeNotification({
                    type: 'call',
                    title: 'Missed Call',
                    message: 'Incoming video call expired.'
                });
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
