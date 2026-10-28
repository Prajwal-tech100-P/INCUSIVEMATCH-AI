"""Transactional email service for InclusiveMatch AI using Brevo SMTP.

Official Brevo SMTP relay documentation:
- Host: smtp-relay.brevo.com
- Port: 587 (STARTTLS) or 465 (SSL)
- Authentication: BREVO_SMTP_LOGIN and BREVO_SMTP_KEY
"""
import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr
from typing import Optional
from flask import current_app

logger = logging.getLogger("EmailService")


def _get_config_value(key: str, default: str = "") -> str:
    """Retrieve config value from Flask current_app if available, fallback to os.environ."""
    if current_app:
        return str(current_app.config.get(key, os.environ.get(key, default)) or "").strip()
    return os.environ.get(key, default).strip()


def send_password_reset_email(
    to_email: str,
    reset_url: str,
    user_name: Optional[str] = None
) -> bool:
    """Send a transactional password-reset email via Brevo SMTP.

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
    sender_email = _get_config_value("BREVO_SENDER_EMAIL", "") or smtp_login
    sender_name = _get_config_value("BREVO_SENDER_NAME", "InclusiveMatch AI")

    # Safety check: if credentials are not configured, log clear diagnostic warning and return False
    if not smtp_login or not smtp_key:
        logger.warning(
            "Brevo SMTP credentials not configured (BREVO_SMTP_LOGIN or BREVO_SMTP_KEY missing). "
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
              <div style="font-size: 26px; font-weight: 800; color: #ffffff; letter-spacing: -0.5px;">
                InclusiveMatch AI
              </div>
              <div style="font-size: 13px; color: #e0e7ff; margin-top: 6px; font-weight: 500;">
                Accessible Communication &amp; Connection Platform
              </div>
            </td>
          </tr>

          <!-- Content Body -->
          <tr>
            <td style="padding: 40px 35px 30px;">
              <h1 style="font-size: 22px; font-weight: 700; color: #0f172a; margin: 0 0 16px 0;">
                Reset Your Password
              </h1>
              <p style="font-size: 15px; line-height: 1.6; color: #475569; margin: 0 0 20px 0;">
                Hello{greeting_name},
              </p>
              <p style="font-size: 15px; line-height: 1.6; color: #475569; margin: 0 0 28px 0;">
                We received a request to reset the password for your InclusiveMatch AI account. Click the button below to choose a new password:
              </p>

              <!-- CTA Button -->
              <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="margin: 0 0 30px 0;">
                <tr>
                  <td align="center">
                    <a href="{reset_url}" target="_blank" rel="noopener" style="display: inline-block; background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%); color: #ffffff; text-decoration: none; padding: 14px 34px; font-size: 15px; font-weight: 700; border-radius: 9999px; box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35);">
                      Reset Password
                    </a>
                  </td>
                </tr>
              </table>

              <!-- Notice Box -->
              <div style="background-color: #f8fafc; border-left: 4px solid #6366f1; border-radius: 8px; padding: 14px 18px; margin: 0 0 24px 0;">
                <p style="font-size: 13px; line-height: 1.5; color: #334155; margin: 0;">
                  <strong>Security Note:</strong> For your protection, this link expires in <strong>30 minutes</strong> and can only be used once.
                </p>
              </div>

              <!-- Fallback Link -->
              <p style="font-size: 13px; line-height: 1.6; color: #64748b; margin: 0 0 16px 0;">
                If the button above does not work, copy and paste this link into your web browser:
              </p>
              <p style="font-size: 12px; line-height: 1.5; color: #4f46e5; word-break: break-all; margin: 0 0 28px 0; background: #f8fafc; padding: 10px 14px; border-radius: 8px; border: 1px solid #e2e8f0;">
                <a href="{reset_url}" target="_blank" rel="noopener" style="color: #4f46e5; text-decoration: underline;">{reset_url}</a>
              </p>

              <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 24px 0;" />

              <p style="font-size: 13px; line-height: 1.5; color: #94a3b8; margin: 0;">
                If you did not request this password reset, you can safely ignore this email. Your current password will remain unchanged unless a reset is completed.
              </p>
            </td>
          </tr>

          <!-- Footer -->
          <tr>
            <td style="background-color: #f8fafc; border-top: 1px solid #e2e8f0; padding: 20px 30px; text-align: center;">
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

    # Assemble MIME multipart alternative
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Reset Your InclusiveMatch AI Password"
    msg["From"] = formataddr((sender_name, sender_email))
    msg["To"] = to_email

    part_text = MIMEText(text_content, "plain", "utf-8")
    part_html = MIMEText(html_content, "html", "utf-8")
    msg.attach(part_text)
    msg.attach(part_html)

    # Dispatch via Brevo SMTP relay
    try:
        if smtp_port == 465:
            with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=12) as server:
                server.login(smtp_login, smtp_key)
                server.sendmail(sender_email, [to_email], msg.as_string())
        else:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=12) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(smtp_login, smtp_key)
                server.sendmail(sender_email, [to_email], msg.as_string())

        logger.info("Successfully sent password-reset email via Brevo SMTP.")
        return True

    except smtplib.SMTPAuthenticationError as e:
        logger.error("Brevo SMTP authentication failed: please check BREVO_SMTP_LOGIN and BREVO_SMTP_KEY.")
        return False
    except smtplib.SMTPException as e:
        logger.error("Brevo SMTP transmission error occurred: %s", type(e).__name__)
        return False
    except Exception as e:
        logger.error("Unexpected error during email dispatch: %s", type(e).__name__)
        return False
