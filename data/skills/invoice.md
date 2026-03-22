# Skill: Invoice Generator

## Trigger patterns
- "generate invoice"
- "new invoice"
- "create receipt"
- "make an invoice"

## Action
Generate a UK-compliant self-employment invoice with sequential numbering.
Endpoint: POST /memory/invoice

## Response template
**Invoice generated:**
- Invoice #: {number}
- Date: {date}
- Amount: £{amount}
- Status: Draft — review and confirm
