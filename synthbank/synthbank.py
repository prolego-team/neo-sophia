import os
import sqlite3

from constructors import make_customer

# count products by customer
# select customers.name, count(products.account_number )
# from customers
# inner join products on customers.guid=products.guid
# group by name;

# get customer name, auto loan accnt number and accnt open date
# select customers.name, products.account_number, auto_loan.account_open_date
# from customers
# inner join products on customers.guid=products.guid
# inner join auto_loan on products.account_number=auto_loan.account_number;

# Make fake customers with attached products
customers = []
for _ in range(1000):
    customer = make_customer()
    customers.append(customer)

DB_FILE = 'synthbank.db'
if os.path.isfile(DB_FILE):
    os.remove(DB_FILE)
con = sqlite3.connect(DB_FILE)
cur = con.cursor()

# Save everything to a database

cur.execute('CREATE TABLE customers(guid NUMERIC, name TEXT, dob TEXT)')
data = [(customer.guid, customer.name, customer.dob) 
        for customer in customers]
cur.executemany('INSERT INTO customers VALUES(?, ?, ?)', data)
con.commit()


cur.execute('CREATE TABLE credit_scores(guid NUMERIC, credit_score NUMERIC)')
data = [(customer.guid, customer.credit_score) for customer in customers]
cur.executemany('INSERT INTO credit_scores VALUES(?, ?)', data)
con.commit()

cur.execute('CREATE TABLE products(guid NUMERIC, account_number TEXT)')
data = [(customer.guid, product.account_number)
        for customer in customers
        for product in customer.products]
cur.executemany('INSERT INTO products VALUES(?, ?)', data)
con.commit()

product_tables = []
type_map = {
    "<class 'int'>": 'NUMERIC',
    "<class 'float'>": 'REAL',
    "<class 'str'>": 'TEXT',
    "<class 'datetime.date'>": 'TEXT'
}
for customer in customers:
    for product in customer.products:
        product_type = product.product
        fields = [field for field in vars(product) if field!='product']
        values = ['"'+str(getattr(product, field))+'"' for field in fields]

        if product_type not in product_tables:
            types = [str(type(getattr(product, field))) for field in vars(product) if field!='product']
            types = [type_map[t] for t in types]
            typed_fields = [f'{field} {typ}' for field,typ in zip(fields,types)]
            cur.execute(f'CREATE TABLE {product_type}({",".join(typed_fields)})')
            product_tables.append(product_type)

        cur.execute(f'INSERT INTO {product_type} VALUES({", ".join(values)})')
con.commit()

