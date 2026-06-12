"""Seed MongoDB with sample banking customer and loan data."""
import os
import sys
import time

MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://mongodb:27017")
MONGODB_DATABASE = os.environ.get("MONGODB_DATABASE", "banking_crm")

CUSTOMERS = [
    {"customer_id": "C-1001", "name": "Alice Tan", "tier": "platinum", "age": 42, "occupation": "CFO", "annual_income": 280000, "credit_score": 780, "account_balance": 125000, "total_deposits": 850000, "relationship_years": 12, "risk_category": "low", "email": "alice.tan@example.com"},
    {"customer_id": "C-1002", "name": "Bob Chen", "tier": "gold", "age": 35, "occupation": "Software Engineer", "annual_income": 120000, "credit_score": 720, "account_balance": 45000, "total_deposits": 320000, "relationship_years": 5, "risk_category": "low", "email": "bob.chen@example.com"},
    {"customer_id": "C-1003", "name": "Carol Wong", "tier": "platinum", "age": 55, "occupation": "Business Owner", "annual_income": 450000, "credit_score": 810, "account_balance": 380000, "total_deposits": 2100000, "relationship_years": 18, "risk_category": "low", "email": "carol.wong@example.com"},
    {"customer_id": "C-1004", "name": "David Lim", "tier": "silver", "age": 28, "occupation": "Marketing Manager", "annual_income": 75000, "credit_score": 650, "account_balance": 12000, "total_deposits": 95000, "relationship_years": 3, "risk_category": "medium", "email": "david.lim@example.com"},
    {"customer_id": "C-1005", "name": "Eva Ng", "tier": "gold", "age": 38, "occupation": "Doctor", "annual_income": 200000, "credit_score": 760, "account_balance": 89000, "total_deposits": 520000, "relationship_years": 8, "risk_category": "low", "email": "eva.ng@example.com"},
    {"customer_id": "C-1006", "name": "Frank Lee", "tier": "silver", "age": 45, "occupation": "Restaurant Owner", "annual_income": 95000, "credit_score": 580, "account_balance": 8500, "total_deposits": 150000, "relationship_years": 6, "risk_category": "high", "email": "frank.lee@example.com"},
    {"customer_id": "C-1007", "name": "Grace Ho", "tier": "diamond", "age": 60, "occupation": "Retired Banker", "annual_income": 350000, "credit_score": 830, "account_balance": 720000, "total_deposits": 4500000, "relationship_years": 25, "risk_category": "low", "email": "grace.ho@example.com"},
    {"customer_id": "C-1008", "name": "Henry Koh", "tier": "gold", "age": 33, "occupation": "Data Scientist", "annual_income": 140000, "credit_score": 740, "account_balance": 52000, "total_deposits": 280000, "relationship_years": 4, "risk_category": "low", "email": "henry.koh@example.com"},
    {"customer_id": "C-1009", "name": "Irene Yap", "tier": "silver", "age": 50, "occupation": "Teacher", "annual_income": 65000, "credit_score": 690, "account_balance": 22000, "total_deposits": 180000, "relationship_years": 15, "risk_category": "medium", "email": "irene.yap@example.com"},
    {"customer_id": "C-1010", "name": "James Ong", "tier": "platinum", "age": 48, "occupation": "Real Estate Developer", "annual_income": 500000, "credit_score": 790, "account_balance": 450000, "total_deposits": 3200000, "relationship_years": 20, "risk_category": "low", "email": "james.ong@example.com"},
]

LOANS = [
    {"customer_id": "C-1001", "loan_id": "L-2001", "type": "mortgage", "amount": 800000, "outstanding": 420000, "interest_rate": 3.2, "term_months": 360, "status": "active", "monthly_payment": 3465, "start_date": "2018-03-15"},
    {"customer_id": "C-1001", "loan_id": "L-2002", "type": "auto", "amount": 45000, "outstanding": 12000, "interest_rate": 4.5, "term_months": 60, "status": "active", "monthly_payment": 840, "start_date": "2022-06-01"},
    {"customer_id": "C-1002", "loan_id": "L-2003", "type": "personal", "amount": 25000, "outstanding": 18000, "interest_rate": 6.8, "term_months": 48, "status": "active", "monthly_payment": 598, "start_date": "2024-01-10"},
    {"customer_id": "C-1003", "loan_id": "L-2004", "type": "business", "amount": 500000, "outstanding": 350000, "interest_rate": 5.0, "term_months": 120, "status": "active", "monthly_payment": 5303, "start_date": "2021-09-20"},
    {"customer_id": "C-1004", "loan_id": "L-2005", "type": "personal", "amount": 15000, "outstanding": 14200, "interest_rate": 9.5, "term_months": 36, "status": "active", "monthly_payment": 481, "start_date": "2025-11-05"},
    {"customer_id": "C-1004", "loan_id": "L-2006", "type": "credit_line", "amount": 10000, "outstanding": 8500, "interest_rate": 18.0, "term_months": 0, "status": "active", "monthly_payment": 250, "start_date": "2024-06-15"},
    {"customer_id": "C-1005", "loan_id": "L-2007", "type": "mortgage", "amount": 600000, "outstanding": 480000, "interest_rate": 3.5, "term_months": 360, "status": "active", "monthly_payment": 2694, "start_date": "2020-02-28"},
    {"customer_id": "C-1006", "loan_id": "L-2008", "type": "business", "amount": 200000, "outstanding": 185000, "interest_rate": 8.0, "term_months": 60, "status": "delinquent", "monthly_payment": 4056, "start_date": "2024-08-01"},
    {"customer_id": "C-1006", "loan_id": "L-2009", "type": "personal", "amount": 30000, "outstanding": 28000, "interest_rate": 12.0, "term_months": 36, "status": "active", "monthly_payment": 997, "start_date": "2025-03-10"},
    {"customer_id": "C-1007", "loan_id": "L-2010", "type": "mortgage", "amount": 1200000, "outstanding": 0, "interest_rate": 2.8, "term_months": 360, "status": "paid_off", "monthly_payment": 0, "start_date": "2005-01-15"},
    {"customer_id": "C-1008", "loan_id": "L-2011", "type": "auto", "amount": 55000, "outstanding": 40000, "interest_rate": 4.2, "term_months": 72, "status": "active", "monthly_payment": 868, "start_date": "2024-04-20"},
    {"customer_id": "C-1010", "loan_id": "L-2012", "type": "business", "amount": 2000000, "outstanding": 1500000, "interest_rate": 4.8, "term_months": 180, "status": "active", "monthly_payment": 15525, "start_date": "2019-07-01"},
]


def seed():
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure

    print(f"[Seed] Connecting to {MONGODB_URI}...")
    for attempt in range(10):
        try:
            client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
            client.server_info()
            break
        except ConnectionFailure:
            print(f"[Seed] Waiting for MongoDB... (attempt {attempt + 1}/10)")
            time.sleep(3)
    else:
        print("[Seed] Could not connect to MongoDB after 10 attempts")
        return

    db = client[MONGODB_DATABASE]

    existing = db.customers.count_documents({})
    if existing > 0:
        print(f"[Seed] Database already has {existing} customers. Skipping seed (idempotent).")
        return

    if CUSTOMERS:
        db.customers.insert_many(CUSTOMERS)
        print(f"[Seed] Inserted {len(CUSTOMERS)} customers into {MONGODB_DATABASE}.customers")

    if LOANS:
        db.loans.insert_many(LOANS)
        print(f"[Seed] Inserted {len(LOANS)} loans into {MONGODB_DATABASE}.loans")

    db.customers.create_index("tier")
    db.customers.create_index("credit_score")
    db.customers.create_index("customer_id", unique=True)
    db.loans.create_index("customer_id")
    db.loans.create_index("loan_id", unique=True)
    print("[Seed] Indexes created. Done!")


if __name__ == "__main__":
    seed()
