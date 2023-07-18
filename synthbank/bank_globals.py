
CUSTOMER_GUIDS = {0}
ACCOUNT_NUMBERS = {100000}

def get_new_account_number():
    new_account = max(ACCOUNT_NUMBERS) + 1
    return new_account

def add_account_number(account_number):
    ACCOUNT_NUMBERS.add(account_number)

def get_new_customer_guid():
    new_guid = max(CUSTOMER_GUIDS) + 1
    return new_guid

def add_customer_guid(guid):
    CUSTOMER_GUIDS.add(guid)