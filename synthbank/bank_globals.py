
CUSTOMER_GUIDS = {0}
ACCOUNT_NUMBERS = {100000}

def get_new_account_number():
    new_account = max(ACCOUNT_NUMBERS) + 1
    ACCOUNT_NUMBERS.add(new_account)
    return new_account


def get_new_customer_guid():
    new_guid = max(CUSTOMER_GUIDS) + 1
    CUSTOMER_GUIDS.add(new_guid)
    return new_guid
