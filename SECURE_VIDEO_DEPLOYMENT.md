# InclusiveMatch AI — Secure Video Call Deployment

## What this version protects

- Flask-Login authentication is required for Socket.IO.
- Every authenticated Socket.IO connection automatically joins only its own private user room.
- Incoming calls are limited to mutual matches.
- Call invitations are stored as short-lived (60 second) records in MongoDB.
- Accept/Decline is protected by the invitation ID and recipient ID.
- WebRTC signaling is allowed only between mutual matches.
- Offer/answer/candidate payloads are type-checked and size-limited.
- Session cookies are HttpOnly, SameSite=Lax, and Secure in production.
- Reverse-proxy HTTPS is supported.
- Security headers are added.
- Socket.IO is not configured with wildcard CORS by default.
- A small HTTP pending-call check keeps the Accept/Decline UI working if Socket.IO briefly reconnects.

## Important: HTTPS

Do NOT use `https://10.x.x.x:5000` with a self-signed certificate as the final deployment.

For local development, `http://127.0.0.1:5000` or `http://localhost:5000` is acceptable for camera/microphone testing on the same computer.

For phone + laptop testing, deploy the application to a real HTTPS hostname. The hosting provider supplies a trusted TLS certificate.

## Important: WebRTC TURN

STUN alone cannot guarantee a call between arbitrary networks. Production should have a TURN service.

Set:

- `WEBRTC_TURN_URL`
- `WEBRTC_TURN_TLS_URL` (recommended if your provider supplies it)
- `WEBRTC_TURN_USERNAME`
- `WEBRTC_TURN_CREDENTIAL`

TURN credentials are delivered to the browser because WebRTC needs them to connect. Prefer short-lived/ephemeral TURN credentials when your TURN provider supports them.

## Render deployment

1. Put the project in a GitHub repository.
2. Create a Render Web Service from that repository.
3. Render can use `render.yaml`.
4. Set `MONGO_URI` to your MongoDB Atlas connection string.
5. Set `SOCKETIO_CORS_ALLOWED_ORIGINS` to the exact HTTPS origin, for example:
   `https://your-app.onrender.com`
6. Add TURN credentials.
7. Deploy.
8. Open the generated HTTPS URL on both laptop and phone.
9. Log in as two matched users.
10. Start a call from one account and Accept on the other.

## Security note

Never put MongoDB passwords, Flask secret keys, TURN credentials, or other secrets in source code or GitHub. Store them as environment variables in Render.

## Testing

The production success criteria are:

- HTTPS opens without a certificate warning.
- Login works.
- Socket.IO connects.
- Incoming call screen appears.
- Accept and Decline work.
- Camera and microphone permission can be granted.
- Video and audio connect on two devices.
- Calls work when the two devices are on different networks.
