from security.rbac import check_access


tests = [
    ("admin", "hr"),
    ("admin", "financial"),
    ("finance", "financial"),
    ("finance", "hr"),
    ("hr", "hr"),
    ("hr", "financial"),
    ("employee", "general"),
    ("employee", "financial"),
    ("viewer", "general"),
    ("viewer", "hr"),
]


for role, category in tests:
    result = check_access(role, category)

    print("-" * 60)
    print(f"Role: {role}")
    print(f"Document Category: {category}")
    print(f"Allowed: {result['allowed']}")
    print(f"Reason: {result['reason']}")