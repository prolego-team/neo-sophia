import sqlite3

from constructors import make_customer

# Make fake customers with attached products
customers = []
for _ in range(1000):
    customer = make_customer()
    customers.append(customer)

con = sqlite3.connect('synthbank.db')
cur = con.cursor()

# Save everything to a database

cur.execute('CREATE TABLE customers(guid, name, dob)')
data = [(customer.guid, customer.name, customer.dob) for customer in customers]
cur.executemany('INSERT INTO customers VALUES(?, ?, ?)', data)
con.commit()


cur.execute('CREATE TABLE credit_scores(guid, credit_score)')
data = [(customer.guid, customer.credit_score) for customer in customers]
cur.executemany('INSERT INTO credit_scores VALUES(?, ?)', data)
con.commit()

cur.execute('CREATE TABLE products(guid, account_number)')
data = [(customer.guid, product.account_number)
        for customer in customers
        for product in customer.products]
cur.executemany('INSERT INTO products VALUES(?, ?)', data)
con.commit()

product_tables = []
for customer in customers:
    for product in customer.products:
        product_type = product.product
        fields = [field for field in vars(product) if field!='product']
        values = [str(getattr(product, field)) for field in fields]

        if product_type not in product_tables:
            cur.execute(f'CREATE TABLE {product_type}({",".join(fields)})')
            product_tables.append(product_type)

        cur.execute(f'INSERT INTO {product_type} VALUES({",".join(values)})')
con.commit()

