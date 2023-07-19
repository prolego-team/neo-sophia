from typing import Tuple

from dataclasses import dataclass, field

from bank_globals import (
    get_new_account_number,
    get_new_customer_guid,
)

@dataclass
class Persona:
    age_range: Tuple[int]
    household_income_range: Tuple[int]
    credit_range: Tuple[int]

@dataclass
class Customer:
    name: str
    dob: str
    credit_score: int
    products: list
    guid: int = field(default_factory=get_new_customer_guid)


@dataclass
class AutoLoan:
    loan_amount: float
    interest_rate: float
    account_number: int = field(default_factory=get_new_account_number)
    product: str = 'auto_loan'
    

@dataclass
class Mortgage:
    loan_amount: float
    interest_rate: float
    account_number: int = field(default_factory=get_new_account_number)
    product = 'mortgage'

@dataclass
class CreditCard:
    credit_limit: int
    interest_rate: float
    account_number: int = field(default_factory=get_new_account_number)
    product = 'credit_card'

@dataclass
class CheckingAccount:
    account_number: int = field(default_factory=get_new_account_number)
    product = 'checking_account'

@dataclass
class SavingsAccount:
    interest_rate: float
    account_number: int = field(default_factory=get_new_account_number)
    product = 'savings_account'