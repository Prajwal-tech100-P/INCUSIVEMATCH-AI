"""Transactional email service for InclusiveMatch AI using Brevo API (HTTPS) and SMTP Relay.

Dual Delivery Support:
1. Primary: Brevo REST API v3 over HTTPS (Port 443) - works on Render free tier where outbound
   SMTP ports 25, 465, and 587 are blocked by cloud firewalls.
2. Fallback: Brevo SMTP Relay over port 587 (STARTTLS) or port 465 (SSL).
"""
import os
import smtplib
import socket
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr
from typing import Optional
import requests
from flask import current_app

logger = logging.getLogger("EmailService")


def _get_config_value(key: str, default: str = "") -> str:
    """Retrieve config value from os.environ first (for dynamic env overrides), fallback to current_app.config."""
    val = os.environ.get(key)
    if val is not None and val != "":
        return val.strip()
    if current_app and current_app.config.get(key):
        return str(current_app.config.get(key)).strip()
    return default.strip()


def _send_via_brevo_api(
    to_email: str,
    user_name: Optional[str],
    sender_name: str,
    sender_email: str,
    subject: str,
    text_content: str,
    html_content: str,
    api_key: str
) -> bool:
    """Dispatch transactional email via Brevo REST API v3 over HTTPS (port 443).

    Bypasses cloud provider port blocks (e.g. Render free tier blocks ports 25, 465, 587).
    """
    url = "https://api.brevo.com/v3/smtp/email"
    headers = {
        "accept": "application/json",
        "api-key": api_key,
        "content-type": "application/json"
    }
    payload = {
        "sender": {
            "name": sender_name,
            "email": sender_email
        },
        "to": [
            {
                "email": to_email,
                "name": user_name or to_email.split("@")[0]
            }
        ],
        "subject": subject,
        "textContent": text_content,
        "htmlContent": html_content
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        if resp.status_code in (200, 201):
            logger.info("Successfully sent password-reset email via Brevo HTTPS REST API.")
            return True
        logger.error(
            "Brevo HTTPS REST API rejected email (status %d): %s",
            resp.status_code,
            resp.text[:200]
        )
        return False
    except requests.Timeout:
        logger.error("Brevo HTTPS REST API connection timed out.")
        return False
    except requests.RequestException as exc:
        logger.error("Brevo HTTPS REST API request failed: %s", type(exc).__name__)
        return False


def send_password_reset_email(
    to_email: str,
    reset_url: str,
    user_name: Optional[str] = None
) -> bool:
    """Send a transactional password-reset email via Brevo HTTPS REST API or SMTP relay.

    Returns True if the email was successfully sent, False otherwise.
    Never exposes reset tokens, SMTP keys, or user credentials in log outputs.
    """
    to_email = (to_email or "").strip()
    if not to_email:
        logger.error("Cannot dispatch reset email: missing recipient email address.")
        return False

    smtp_host = _get_config_value("BREVO_SMTP_HOST", "smtp-relay.brevo.com")
    raw_port = _get_config_value("BREVO_SMTP_PORT", "587")
    try:
        smtp_port = int(raw_port)
    except (ValueError, TypeError):
        smtp_port = 587

    smtp_login = _get_config_value("BREVO_SMTP_LOGIN", "")
    smtp_key = _get_config_value("BREVO_SMTP_KEY", "")
    api_key = _get_config_value("BREVO_API_KEY", "") or smtp_key
    sender_email = _get_config_value("BREVO_SENDER_EMAIL", "") or smtp_login
    sender_name = _get_config_value("BREVO_SENDER_NAME", "InclusiveMatch AI")

    # Safety check: if no credentials are configured, log warning and return False
    if not api_key and (not smtp_login or not smtp_key):
        logger.warning(
            "Brevo credentials not configured (BREVO_API_KEY or BREVO_SMTP_KEY missing). "
            "Email dispatch skipped."
        )
        return False

    if not sender_email:
        logger.warning(
            "Brevo sender email not configured (BREVO_SENDER_EMAIL missing). Email dispatch skipped."
        )
        return False

    greeting_name = f" {user_name}" if user_name else ""

    # Plain text alternative
    text_content = (
        f"Hello{greeting_name},\n\n"
        "We received a request to reset the password for your InclusiveMatch AI account.\n\n"
        "Click the link below to choose a new password:\n"
        f"{reset_url}\n\n"
        "For your security, this link expires in 30 minutes and can only be used once.\n\n"
        "If you did not request this change, you can safely ignore this email. "
        "Your current password will remain unchanged unless a reset is completed.\n\n"
        "Regards,\n"
        "InclusiveMatch AI Team\n"
        "https://incusivematch-ai.onrender.com"
    )

    # Rich, responsive HTML version matching InclusiveMatch AI theme
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Reset Your Password - InclusiveMatch AI</title>
</head>
<body style="margin: 0; padding: 0; background-color: #f1f5f9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; color: #1e293b;">
  <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background-color: #f1f5f9; padding: 40px 15px;">
    <tr>
      <td align="center">
        <!-- Main Card -->
        <table role="presentation" width="100%" style="max-width: 580px; background-color: #ffffff; border-radius: 20px; box-shadow: 0 10px 25px rgba(0,0,0,0.06); overflow: hidden; border: 1px solid #e2e8f0;">
          <!-- Header -->
          <tr>
            <td style="background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); padding: 35px 30px; text-align: center;">
              <h1 style="color: #ffffff; margin: 0; font-size: 24px; font-weight: 700; letter-spacing: -0.5px;">
                InclusiveMatch AI
              </h1>
              <p style="color: #e0e7ff; margin: 8px 0 0 0; font-size: 13px; font-weight: 400;">
                Accessible &bull; Secure &bull; Connected
              </p>
            </td>
          </tr>

          <!-- Content Body -->
          <tr>
            <td style="padding: 35px 30px;">
              <h2 style="font-size: 18px; font-weight: 600; color: #0f172a; margin-top: 0; margin-bottom: 12px;">
                Password Reset Request
              </h2>
              <p style="font-size: 14px; line-height: 1.6; color: #475569; margin: 0 0 20px 0;">
                Hello{greeting_name},
              </p>
              <p style="font-size: 14px; line-height: 1.6; color: #475569; margin: 0 0 25px 0;">
                We received a request to reset the password associated with your InclusiveMatch AI account. If you made this request, please click the button below to choose your new password.
              </p>

              <!-- Reset Button -->
              <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="margin-bottom: 25px;">
                <tr>
                  <td align="center">
                    <a href="{reset_url}" target="_blank" style="display: inline-block; background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); color: #ffffff; font-size: 14px; font-weight: 600; text-decoration: none; padding: 13px 32px; border-radius: 10px; box-shadow: 0 4px 12px rgba(99, 102, 241, 0.35); text-align: center;">
                      Reset Password
                    </a>
                  </td>
                </tr>
              </table>

              <!-- Expiry & Safety Notice -->
              <div style="background-color: #f8fafc; border-left: 4px solid #6366f1; padding: 12px 16px; border-radius: 6px; margin-bottom: 25px;">
                <p style="font-size: 12px; line-height: 1.5; color: #64748b; margin: 0;">
                  <strong>Security Note:</strong> This password reset link is strictly valid for <strong>30 minutes</strong> and can only be used once.
                </p>
              </div>

              <!-- Fallback Direct Link -->
              <p style="font-size: 12px; line-height: 1.5; color: #64748b; margin: 0 0 8px 0;">
                If the button above does not work, copy and paste this URL into your browser:
              </p>
              <p style="font-size: 11px; word-break: break-all; color: #4f46e5; margin: 0 0 25px 0; background: #eef2ff; padding: 10px; border-radius: 6px; font-family: monospace;">
                {reset_url}
              </p>

              <p style="font-size: 13px; line-height: 1.5; color: #64748b; margin: 0;">
                If you did not request this password reset, you can safely ignore this email. Your current password will remain unchanged unless a reset is completed.
              </p>
            </td>
          </tr>

          <!-- Footer -->
          <tr>
            <td style="background-color: #f8fafc; padding: 20px 30px; text-align: center; border-top: 1px solid #f1f5f9;">
              <p style="font-size: 12px; color: #94a3b8; margin: 0 0 6px 0;">
                &copy; 2026 InclusiveMatch AI. All rights reserved.
              </p>
              <p style="font-size: 11px; color: #cbd5e1; margin: 0;">
                Empowering the Deaf and Hard-of-Hearing community with accessible, secure AI communication.
              </p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""

    subject = "Reset Your InclusiveMatch AI Password"

    # 1. Primary Path: Brevo REST API v3 over HTTPS (Port 443)
    # Required for cloud hosts like Render free tier where outbound SMTP ports 25, 465, 587 are blocked.
    if api_key:
        api_sent = _send_via_brevo_api(
            to_email=to_email,
            user_name=user_name,
            sender_name=sender_name,
            sender_email=sender_email,
            subject=subject,
            text_content=text_content,
            html_content=html_content,
            api_key=api_key
        )
        if api_sent:
            return True
        logger.info("Brevo HTTPS API delivery did not succeed. Attempting SMTP relay fallback...")

    # 2. Fallback Path: Brevo SMTP Relay over port 587 (STARTTLS) or port 465 (SSL)
    if smtp_login and smtp_key:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = formataddr((sender_name, sender_email))
        msg["To"] = to_email

        part_text = MIMEText(text_content, "plain", "utf-8")
        part_html = MIMEText(html_content, "html", "utf-8")
        msg.attach(part_text)
        msg.attach(part_html)

        try:
            if smtp_port == 465:
                with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=10) as server:
                    server.login(smtp_login, smtp_key)
                    server.sendmail(sender_email, [to_email], msg.as_string())
            else:
                with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
                    server.ehlo()
                    server.starttls()
                    server.ehlo()
                    server.login(smtp_login, smtp_key)
                    server.sendmail(sender_email, [to_email], msg.as_string())

            logger.info("Successfully sent password-reset email via Brevo SMTP relay.")
            return True

        except smtplib.SMTPAuthenticationError:
            logger.error("Brevo SMTP authentication failed: please check BREVO_SMTP_LOGIN and BREVO_SMTP_KEY.")
            return False
        except (TimeoutError, socket.timeout):
            logger.error(
                "Brevo SMTP connection timed out. On Render free instances, outbound SMTP ports (25, 465, 587) "
                "are blocked by the platform firewall. Please configure BREVO_API_KEY in Render to send via HTTPS (port 443)."
            )
            return False
        except smtplib.SMTPException as exc:
            logger.error("Brevo SMTP transmission error occurred: %s", type(exc).__name__)
            return False
        except Exception as exc:
            logger.error("Unexpected error during email dispatch: %s", type(exc).__name__)
            return False

    return False
