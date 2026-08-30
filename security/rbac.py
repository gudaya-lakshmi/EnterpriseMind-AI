from typing import Dict, List


ROLE_PERMISSIONS: Dict[str, List[str]] = {
    "admin": ["all"],
    "finance": ["financial", "general"],
    "hr": ["hr", "general"],
    "employee": ["general"],
    "viewer": ["general"],
}


def get_allowed_categories(role: str) -> List[str]:
    """
    Return document categories that a role is allowed to access.
    """

    role = role.lower().strip()

    return ROLE_PERMISSIONS.get(role, [])


def can_access_document(role: str, document_category: str) -> bool:
    """
    Check whether a user role can access a document category.
    """

    role = role.lower().strip()
    document_category = document_category.lower().strip()

    allowed_categories = get_allowed_categories(role)

    if "all" in allowed_categories:
        return True

    return document_category in allowed_categories


def check_access(role: str, document_category: str) -> Dict:
    """
    Return a structured RBAC decision.
    """

    allowed = can_access_document(role, document_category)

    if allowed:
        reason = (
            f"Role '{role}' is authorized to access "
            f"'{document_category}' documents."
        )
    else:
        reason = (
            f"Role '{role}' is not authorized to access "
            f"'{document_category}' documents."
        )

    return {
        "role": role,
        "document_category": document_category,
        "allowed": allowed,
        "reason": reason,
    }