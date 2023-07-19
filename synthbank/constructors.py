import random
import datetime

import numpy as np

from components import (
    Persona, Customer,
    AutoLoan, Mortgage, CreditCard, CheckingAccount, SavingsAccount
)

persona_data = [
    (0.02, Persona(
        age_range=(55, 85),
        household_income_range=(250000, 750000),
        credit_range=(750, 850),
    )),
    (0.20, Persona(
        age_range=(55, 85),
        household_income_range=(70000, 150000),
        credit_range=(550, 800),
    )),
    (0.20, Persona(
        age_range=(35, 55),
        household_income_range=(150000, 350000),
        credit_range=(650, 810),
    )),
    (0.30, Persona(
        age_range=(35, 55),
        household_income_range=(60000, 150000),
        credit_range=(500, 800),
    )),
    (0.10, Persona(
        age_range=(25,35),
        household_income_range=(100000, 250000),
        credit_range=(600,825)
    )),
    (0.10, Persona(
        age_range=(25,35),
        household_income_range=(35000, 75000),
        credit_range=(500,750)
    )),
    (0.08, Persona(
        age_range=(18,25),
        household_income_range=(15000, 65000),
        credit_range=(500,750)
    )),
]
persona_probs, personas = zip(*persona_data)
persona_probs_cumulative = np.cumsum(persona_probs)/sum(persona_probs)

names_first = []
names_last = []
with open('synthbank/census_names_1990/original/dist.all.last', 'r') as f:
    for line in f:
        names_last.append(line.split(' ')[0])
with open('synthbank/census_names_1990/original/dist.female.first', 'r') as f:
    for line in f:
        names_first.append(line.split(' ')[0])
with open('synthbank/census_names_1990/original/dist.male.first', 'r') as f:
    for line in f:
        names_first.append(line.split(' ')[0])


products = (
    'auto_loan',
    'mortgage',
    'credit_card',
    'checking_account',
    'savings_account'
)

credit_score_range = (500, 850)
today = datetime.date.today()
random.seed(0)

def get_name():
    name_last = random.choice(names_last)
    name_first = random.choice(names_first)
    return name_first+' '+name_last

def get_dob(age_range):
    age = random.randint(*age_range)
    dob = today - datetime.timedelta(days=age*365.25)
    return dob

def get_credit_score(credit_score_range):
    return random.randint(*credit_score_range)

def make_auto_loan(credit_score):
    interest_rate_range = (0.035, 0.10)
    loan_amount_range = (10000, 80000)

    credit_multiplier = (credit_score_range[1] - credit_score)/(credit_score_range[1] - credit_score_range[0])
    interest_rate = interest_rate_range[0] + credit_multiplier*(interest_rate_range[1] - interest_rate_range[0])
    loan_amount = loan_amount_range[0] + random.random()*(loan_amount_range[1] - loan_amount_range[0])

    loan_amount = round(loan_amount, 0)
    interest_rate = round(interest_rate, 3)
    return AutoLoan(loan_amount, interest_rate)

def make_morgage(credit_score):
    interest_rate_range = (0.045, 0.10)
    loan_amount_range = (200000, 1500000)

    credit_multiplier = (credit_score_range[1] - credit_score)/(credit_score_range[1] - credit_score_range[0])
    interest_rate = interest_rate_range[0] + credit_multiplier*(interest_rate_range[1] - interest_rate_range[0])
    loan_amount = loan_amount_range[0] + random.random()*(loan_amount_range[1] - loan_amount_range[0])

    loan_amount = round(loan_amount, 0)
    interest_rate = round(interest_rate, 3)
    return Mortgage(loan_amount, interest_rate)

def make_credit_card(credit_score):
    interest_rate_range = (0.15, 0.30)
    credit_limit_range = (1000, 50000)

    credit_multiplier = (credit_score_range[1] - credit_score)/(credit_score_range[1] - credit_score_range[0])
    interest_rate = interest_rate_range[0] + credit_multiplier*(interest_rate_range[1] - interest_rate_range[0])
    credit_limit = credit_limit_range[0] + random.random()*(credit_limit_range[1] - credit_limit_range[0])

    credit_limit = round(credit_limit-(credit_limit%1000), 0)
    interest_rate = round(interest_rate, 2)
    return CreditCard(credit_limit, interest_rate)

def make_checking_account():
    return CheckingAccount()

def make_savings_account():
    interest_rate_range = (0.005, 0.015)

    interest_rate = interest_rate_range[0] + random.random()*(interest_rate_range[1] - interest_rate_range[0])
    interest_rate = round(interest_rate, 3)
    return SavingsAccount(interest_rate)

def make_customer():
    # Sample a persona
    random_thresh = random.random()
    for cumul_prob,persona in zip(persona_probs_cumulative, personas):
        if cumul_prob>=random_thresh:
            break

    name = get_name()
    dob = get_dob(persona.age_range)
    credit_score = get_credit_score(persona.credit_range)
    customer = Customer(name, dob, credit_score, [])

    n_proucts = random.randint(1,4)
    customer_products = [random.choice(products) for _ in range(n_proucts)]
    for product in customer_products:
        if product=='auto_loan':
            customer.products.append(make_auto_loan(credit_score))
        elif product=='mortgage':
            customer.products.append(make_morgage(credit_score))
        elif product=='credit_card':
            customer.products.append(make_credit_card(credit_score))
        elif product=='checking_account':
            customer.products.append(make_checking_account())
        elif product=='savings_account':
            customer.products.append(make_savings_account())
    
    return customer