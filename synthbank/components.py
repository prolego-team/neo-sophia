from typing import Tuple

from dataclasses import dataclass, field

from bank_globals import (
    get_new_account_number, add_account_number,
    get_new_customer_guid, add_customer_guid
)

@dataclass
class Persona:
    age_range: Tuple[int]
    household_income_range: Tuple[int]
    credit_range: Tuple[int]

@dataclass
class Customer:
    dob: str
    credit_score: int
    products: list
    guid: int = field(default_factory=get_new_customer_guid)

    def __post_init__(self):
        add_customer_guid(self.guid)

@dataclass
class AccountProudct:
    def __post_init__(self):
        add_account_number(self.account_number)

@dataclass
class AutoLoan(AccountProudct):
    loan_amount: float
    interest_rate: float
    account_number: int = field(default_factory=get_new_account_number)
    product: str = 'auto_loan'
    

@dataclass
class Mortgage(AccountProudct):
    loan_amount: float
    interest_rate: float
    account_number: int = field(default_factory=get_new_account_number)
    product = 'mortgage'

@dataclass
class CreditCard(AccountProudct):
    credit_limit: int
    interest_rate: float
    account_number: int = field(default_factory=get_new_account_number)
    product = 'credit_card'

@dataclass
class CheckingAccount(AccountProudct):
    account_number: int = field(default_factory=get_new_account_number)
    product = 'checking_account'

@dataclass
class SavingsAccount(AccountProudct):
    interest_rate: float
    account_number: int = field(default_factory=get_new_account_number)
    product = 'savings_account'