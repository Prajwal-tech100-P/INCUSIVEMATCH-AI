"""Legal, Privacy, and User Protection Routes for InclusiveMatch AI.

Implements all statutory and informational legal pages under Indian legal framework:
- Digital Personal Data Protection Act, 2023 (DPDP Act, 2023)
- Information Technology Act, 2000 & IT (Intermediary Guidelines) Rules, 2021
- Rights of Persons with Disabilities Act, 2016 (RPwD Act)
"""

import uuid
from datetime import datetime, timedelta
from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify,
    Response,
    current_app,
)
from flask_login import current_user, login_required
from extensions import mongo

legal_bp = Blueprint("legal", __name__)

# Canonical policy versions for audit trail and consent tracking
POLICY_VERSIONS = {
    "privacy_policy": "2026.1",
    "terms_conditions": "2026.1",
    "community_guidelines": "2026.1",
    "user_safety": "2026.1",
    "cookie_policy": "2026.1",
    "ai_transparency": "2026.1",
    "accessibility_statement": "2026.1",
    "data_rights": "2026.1",
    "contact_grievance": "2026.1",
    "last_updated": "27 October 2026",
}


@legal_bp.app_context_processor
def inject_policy_meta():
    """Inject policy versions and metadata globally into templates."""
    return {"legal_meta": POLICY_VERSIONS}


# ─────────────────────────────────────────────────────────────────────────────
# 1. LEGAL DOCUMENT ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@legal_bp.route("/privacy")
def privacy_policy():
    """Privacy Policy & Notice of Digital Personal Data Processing."""
    return render_template(
        "legal/privacy_policy.html",
        active_doc="privacy",
        versions=POLICY_VERSIONS,
    )


@legal_bp.route("/terms")
def terms_conditions():
    """Terms and Conditions of Service (Draft for Legal Review)."""
    return render_template(
        "legal/terms_conditions.html",
        active_doc="terms",
        versions=POLICY_VERSIONS,
    )


@legal_bp.route("/community-guidelines")
def community_guidelines():
    """Community Guidelines and Dating Etiquette."""
    return render_template(
        "legal/community_guidelines.html",
        active_doc="guidelines",
        versions=POLICY_VERSIONS,
    )


@legal_bp.route("/safety")
def user_safety():
    """User Safety and Dating Protection Policy."""
    return render_template(
        "legal/user_safety.html",
        active_doc="safety",
        versions=POLICY_VERSIONS,
    )


@legal_bp.route("/cookie-policy")
def cookie_policy():
    """Cookie and Local Storage Policy."""
    return render_template(
        "legal/cookie_policy.html",
        active_doc="cookies",
        versions=POLICY_VERSIONS,
    )


@legal_bp.route("/ai-transparency")
def ai_transparency():
    """AI Transparency and Algorithmic Decision Notice."""
    return render_template(
        "legal/ai_transparency.html",
        active_doc="ai",
        versions=POLICY_VERSIONS,
    )


@legal_bp.route("/accessibility")
def accessibility_statement():
    """Accessibility Statement and Assistive Communication Standards."""
    return render_template(
        "legal/accessibility_statement.html",
        active_doc="accessibility",
        versions=POLICY_VERSIONS,
    )


@legal_bp.route("/data-rights")
def data_rights():
    """Account Deletion, Retention, and Data Subject Rights Policy."""
    return render_template(
        "legal/data_rights.html",
        active_doc="data_rights",
        versions=POLICY_VERSIONS,
    )


@legal_bp.route("/contact-grievance", methods=["GET"])
def contact_grievance():
    """Contact, Grievance Redressal, and Statutory Inquiries."""
    user_grievances = []
    if current_user.is_authenticated:
        user_grievances = list(mongo.db.grievances.find({
            "$or": [
                {"user_id": str(current_user.id)},
                {"email": current_user.email}
            ]
        }).sort("created_at", -1))
        for g in user_grievances:
            dt = g.get("created_at")
            if isinstance(dt, datetime):
                g["formatted_date"] = (dt + timedelta(hours=5, minutes=30)).strftime("%d %b %Y, %I:%M %p")
            else:
                g["formatted_date"] = str(dt) if dt else "N/A"
            rdt = g.get("resolved_at")
            if isinstance(rdt, datetime):
                g["formatted_resolved_date"] = (rdt + timedelta(hours=5, minutes=30)).strftime("%d %b %Y, %I:%M %p")
            else:
                g["formatted_resolved_date"] = ""

    return render_template(
        "legal/contact_grievance.html",
        active_doc="grievance",
        versions=POLICY_VERSIONS,
        user_grievances=user_grievances,
    )


@legal_bp.route("/terms-of-use", methods=["GET"])
def terms_of_use_alias():
    """Alias for Terms and Conditions."""
    return redirect(url_for("legal.terms_conditions"))


@legal_bp.route("/contact", methods=["GET"])
@legal_bp.route("/support", methods=["GET"])
@legal_bp.route("/grievance", methods=["GET"])
def contact_support_alias():
    """Redirect aliases to the unified Contact & Grievance portal."""
    return redirect(url_for("legal.contact_grievance"))


# ─────────────────────────────────────────────────────────────────────────────
# 2. GRIEVANCE & PRIVACY REQUEST SUBMISSION (PHASE 10)
# ─────────────────────────────────────────────────────────────────────────────

VALID_GRIEVANCE_CATEGORIES = {
    "data_access": "Data Access Request (DPDP Act)",
    "data_correction": "Data Correction or Updating (DPDP Act)",
    "data_erasure": "Data Deletion / Erasure Request",
    "privacy_complaint": "Privacy Complaint or Violation",
    "safety_misconduct": "User Safety, Harassment or Misconduct Report",
    "accessibility_barrier": "Accessibility Barrier Report",
    "general_legal": "General Legal or Terms Inquiry",
}


@legal_bp.route("/contact-grievance/submit", methods=["POST"])
def submit_grievance():
    """Validate and record user grievances or data rights requests in MongoDB."""
    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip().lower()
    category = request.form.get("category", "").strip()
    subject = request.form.get("subject", "").strip()
    message = request.form.get("message", "").strip()

    # If user is authenticated, pre-fill / corroborate user details
    user_id = str(current_user.id) if current_user.is_authenticated else None
    if current_user.is_authenticated and not email:
        email = getattr(current_user, "email", "")
    if current_user.is_authenticated and not name:
        name = getattr(current_user, "name", "")

    # Input validation
    errors = []
    if not name or len(name) < 2:
        errors.append("Please provide a valid full name.")
    if not email or "@" not in email or "." not in email:
        errors.append("Please provide a valid email address for grievance correspondence.")
    if category not in VALID_GRIEVANCE_CATEGORIES:
        errors.append("Please select a valid grievance or request category.")
    if not subject or len(subject) < 3:
        errors.append("Please enter a descriptive subject (at least 3 characters).")
    if not message or len(message) < 15:
        errors.append("Please provide sufficient details in your message (at least 15 characters).")

    if errors:
        for err in errors:
            flash(err, "danger")
        return redirect(url_for("legal.contact_grievance"))

    # Generate unique, audit-compliant tracking ID (format: GRV-YYYYMMDD-XXXX)
    date_str = datetime.utcnow().strftime("%Y%m%d")
    short_uuid = uuid.uuid4().hex[:6].upper()
    tracking_id = f"GRV-{date_str}-{short_uuid}"

    grievance_doc = {
        "tracking_id": tracking_id,
        "user_id": user_id,
        "name": name,
        "email": email,
        "category": category,
        "category_label": VALID_GRIEVANCE_CATEGORIES[category],
        "subject": subject,
        "message": message,
        "status": "pending",
        "ip_address": request.headers.get("X-Forwarded-For", request.remote_addr),
        "user_agent": request.headers.get("User-Agent", "")[:250],
        "created_at": datetime.utcnow(),
        "resolved_at": None,
        "resolution_notes": None,
    }

    try:
        mongo.db.grievances.insert_one(grievance_doc)
        current_app.logger.info(
            f"Grievance filed successfully. Tracking ID: {tracking_id}, Category: {category}"
        )
        flash(
            f"Your request has been registered under Tracking ID: {tracking_id}. "
            "Our Grievance Redressal mechanism will acknowledge your communication "
            "within 24 hours and address it within statutory timeframes.",
            "success",
        )
    except Exception as e:
        current_app.logger.error(f"Failed to record grievance in MongoDB: {e}")
        flash(
            "An error occurred while saving your request. Please try again or email the Grievance Officer directly.",
            "danger",
        )

    return redirect(url_for("legal.contact_grievance"))


@legal_bp.route("/contact-grievance/track", methods=["GET", "POST"])
def track_grievance():
    """Allow users to securely check the status of their filed grievance using Tracking ID and Email."""
    tracking_result = None
    searched = False

    if request.method == "POST":
        searched = True
        tracking_id = request.form.get("tracking_id", "").strip().upper()
        email = request.form.get("email", "").strip().lower()

        if not tracking_id or not email:
            flash("Please enter both your Tracking ID and registered email address.", "danger")
        else:
            doc = mongo.db.grievances.find_one({"tracking_id": tracking_id, "email": email})
            if doc:
                dt = doc.get("created_at")
                if isinstance(dt, datetime):
                    doc["formatted_date"] = (dt + timedelta(hours=5, minutes=30)).strftime("%d %b %Y, %I:%M %p")
                else:
                    doc["formatted_date"] = str(dt) if dt else "N/A"
                rdt = doc.get("resolved_at")
                if isinstance(rdt, datetime):
                    doc["formatted_resolved_date"] = (rdt + timedelta(hours=5, minutes=30)).strftime("%d %b %Y, %I:%M %p")
                else:
                    doc["formatted_resolved_date"] = ""
                tracking_result = doc
            else:
                flash("No grievance found matching that Tracking ID and email address.", "warning")

    user_grievances = []
    if current_user.is_authenticated:
        user_grievances = list(mongo.db.grievances.find({
            "$or": [
                {"user_id": str(current_user.id)},
                {"email": current_user.email}
            ]
        }).sort("created_at", -1))
        for g in user_grievances:
            dt = g.get("created_at")
            if isinstance(dt, datetime):
                g["formatted_date"] = (dt + timedelta(hours=5, minutes=30)).strftime("%d %b %Y, %I:%M %p")
            else:
                g["formatted_date"] = str(dt) if dt else "N/A"
            rdt = g.get("resolved_at")
            if isinstance(rdt, datetime):
                g["formatted_resolved_date"] = (rdt + timedelta(hours=5, minutes=30)).strftime("%d %b %Y, %I:%M %p")
            else:
                g["formatted_resolved_date"] = ""

    return render_template(
        "legal/contact_grievance.html",
        active_doc="grievance",
        versions=POLICY_VERSIONS,
        user_grievances=user_grievances,
        tracking_result=tracking_result,
        searched=searched,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. SELF-SERVICE DATA EXPORT (DPDP ACT SECTION 11 RIGHT TO ACCESS)
# ─────────────────────────────────────────────────────────────────────────────

@legal_bp.route("/data-rights/export-my-data", methods=["GET"])
@login_required
def export_my_data():
    """Provide authenticated users with a machine-readable JSON export of their personal data."""
    try:
        from bson import ObjectId
        import json

        user_doc = mongo.db.users.find_one({"_id": ObjectId(current_user.id)}) or {}

        # Strip password hash and internal security salts
        safe_user = dict(user_doc)
        if "_id" in safe_user:
            safe_user["_id"] = str(safe_user["_id"])
        safe_user.pop("password", None)

        # Retrieve user's messages (sent & received metadata)
        messages_cursor = mongo.db.messages.find(
            {"$or": [{"sender_id": current_user.id}, {"receiver_id": current_user.id}]}
        ).sort("timestamp", 1)
        messages_list = []
        for m in messages_cursor:
            messages_list.append({
                "sender_id": m.get("sender_id"),
                "receiver_id": m.get("receiver_id"),
                "text": m.get("text"),
                "timestamp": str(m.get("timestamp")),
                "is_read": m.get("is_read"),
            })

        # Retrieve active restrictions initiated
        restrictions = list(mongo.db.user_restrictions.find(
            {"restrictor_id": current_user.id},
            {"_id": 0, "restricted_user_id": 1, "reason": 1, "created_at": 1}
        ))
        for r in restrictions:
            if "created_at" in r:
                r["created_at"] = str(r["created_at"])

        export_payload = {
            "platform": "InclusiveMatch AI",
            "statutory_basis": "Digital Personal Data Protection Act, 2023 (Section 11)",
            "exported_at": datetime.utcnow().isoformat() + "Z",
            "account_profile": safe_user,
            "private_messages_count": len(messages_list),
            "private_messages": messages_list,
            "user_restrictions": restrictions,
        }

        json_bytes = json.dumps(export_payload, indent=2, default=str)
        filename = f"inclusivematch_data_export_{current_user.id}_{datetime.utcnow().strftime('%Y%m%d')}.json"

        return Response(
            json_bytes,
            mimetype="application/json",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except Exception as e:
        current_app.logger.error(f"Data export failed for user {current_user.id}: {e}")
        flash("Failed to generate data export. Please try again or contact the Grievance Officer.", "danger")
        return redirect(url_for("profile.my_profile"))
