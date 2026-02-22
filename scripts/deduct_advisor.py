from __future__ import annotations

from common.self_emp_advisor import advise, load_expenses, load_income_total

income = load_income_total("/data/self-emp/Accounting/income.csv")
expenses = load_expenses("/data/self-emp/Accounting/expenses.log")
for line in advise(income_total=income, expenses_lines=expenses):
    print(line)
