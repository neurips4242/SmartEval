"""
Generate various contract PDFs for testing agentic_implementation.py
"""

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def create_contract(filename, title, content):
    """Create a PDF contract"""
    doc = SimpleDocTemplate(
        str(filename), pagesize=letter, topMargin=0.5 * inch, bottomMargin=0.5 * inch
    )
    story = []
    styles = getSampleStyleSheet()
    style = ParagraphStyle(
        name="Body", parent=styles["Normal"], fontSize=10, leading=12
    )
    title_style = ParagraphStyle(
        name="Title",
        parent=styles["Heading1"],
        fontSize=13,
        leading=15,
        spaceAfter=10,
        alignment=1,
    )

    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(content, style))
    doc.build(story)
    print(f"✓ {filename}")


contracts_dir = Path(__file__).parent / "contracts"
contracts_dir.mkdir(exist_ok=True)

# 1. EMPLOYMENT CONTRACT
employment_content = """
<b>EMPLOYMENT AGREEMENT</b><br/>
Date: January 15, 2025<br/><br/>

<b>BETWEEN:</b> TechCorp Industries, Inc., a Delaware corporation ("Employer")<br/>
<b>AND:</b> Robert Johnson, residing at 456 Oak Avenue, San Francisco, CA 94102 ("Employee")<br/><br/>

<b>1. POSITION AND DUTIES</b><br/>
Employee shall serve as Senior Software Engineer, reporting to Chief Technology Officer. Responsibilities include developing enterprise-level applications, mentoring junior developers, and architectural decisions.<br/><br/>

<b>2. TERM</b><br/>
Commencement: February 1, 2025<br/>
Duration: Three (3) years from commencement date<br/><br/>

<b>3. COMPENSATION</b><br/>
Base Salary: USD 150,000 per annum, payable monthly at USD 12,500<br/>
Performance Bonus: Up to USD 25,000 annually based on objectives<br/>
Stock Options: 5,000 options vesting over 4 years with 1-year cliff<br/>
Health Insurance: Full coverage for employee and family<br/>
Retirement: 401(k) with 5% employer match<br/>
Paid Time Off: 20 vacation days, 10 sick days, 6 company holidays annually<br/><br/>

<b>4. TERMINATION</b><br/>
At-Will Employment: Either party may terminate with 30 days written notice<br/>
Severance: 3 months base salary upon termination without cause<br/>
Cause: Breach of confidentiality, gross negligence, or policy violation<br/><br/>

<b>5. CONFIDENTIALITY AND NON-COMPETE</b><br/>
Employee agrees to maintain confidentiality of all proprietary information.<br/>
Non-Compete Period: 12 months post-employment within 100-mile radius of headquarters<br/><br/>

Employee: ___________________  Date: _____________<br/>
Employer: ___________________  Date: _____________
"""

create_contract(
    contracts_dir / "Employment_Contract.pdf",
    "EMPLOYMENT AGREEMENT",
    employment_content,
)

# 2. SALES AGREEMENT
sales_content = """
<b>SALES AND PURCHASE AGREEMENT</b><br/>
Date: December 20, 2024<br/><br/>

<b>SELLER:</b> TechGear Manufacturing LLC, located at 789 Industrial Park, Austin, TX 78701<br/>
<b>BUYER:</b> RetailChain Corp, located at 321 Commerce Street, New York, NY 10001<br/><br/>

<b>1. GOODS</b><br/>
Product: Industrial Robotic Arms, Model X-500<br/>
Quantity: 50 units<br/>
Unit Price: USD 85,000<br/>
Total Purchase Price: USD 4,250,000<br/><br/>

<b>2. DELIVERY</b><br/>
Delivery Location: Buyer's warehouse in New Jersey<br/>
Delivery Date: January 31, 2025<br/>
Shipping Method: Ground transportation with insurance coverage<br/>
Delivery Cost: USD 50,000 (borne by Seller)<br/><br/>

<b>3. PAYMENT TERMS</b><br/>
30% down payment (USD 1,275,000) upon signing<br/>
40% upon production completion (USD 1,700,000)<br/>
30% upon delivery and inspection (USD 1,275,000)<br/><br/>

<b>4. WARRANTY</b><br/>
Seller warrants goods free from defects for 24 months from delivery<br/>
Manufacturer warranty transferred to Buyer<br/>
Defective units subject to replacement or repair<br/><br/>

<b>5. INSPECTION AND ACCEPTANCE</b><br/>
Buyer has 15 days to inspect goods<br/>
Acceptance upon satisfactory functional testing<br/>
Any defects must be reported within 30 days<br/><br/>

<b>6. TERMINATION</b><br/>
Either party may terminate if payment not received by due date<br/>
Termination Fee: 10% of undelivered goods value<br/><br/>

Seller: ___________________  Date: _____________<br/>
Buyer: ___________________  Date: _____________
"""

create_contract(
    contracts_dir / "Sales_Agreement.pdf", "SALES AND PURCHASE AGREEMENT", sales_content
)

# 3. LOAN AGREEMENT
loan_content = """
<b>PROMISSORY NOTE AND LOAN AGREEMENT</b><br/>
Date: January 10, 2025<br/><br/>

<b>LENDER:</b> Capital Finance Group, LLC, located at 555 Finance Plaza, Boston, MA 02101<br/>
<b>BORROWER:</b> GreenTech Solutions, Inc., a Delaware corporation<br/><br/>

<b>1. PRINCIPAL AMOUNT</b><br/>
Loan Amount: USD 500,000<br/>
Disbursement Date: February 1, 2025<br/>
Loan Purpose: Working capital and equipment purchase<br/><br/>

<b>2. INTEREST RATE AND FEES</b><br/>
Annual Interest Rate: 7.5% fixed<br/>
Origination Fee: 2% of principal (USD 10,000)<br/>
Late Payment Fee: 5% of overdue amount<br/><br/>

<b>3. REPAYMENT SCHEDULE</b><br/>
Term: 5 years (60 months)<br/>
Monthly Payment: USD 9,934<br/>
First Payment Due: March 1, 2025<br/>
Payment Method: Automatic ACH debit from business checking account<br/><br/>

<b>4. PREPAYMENT</b><br/>
Borrower may prepay without penalty<br/>
Early prepayment discounts: 3% if paid within 12 months<br/><br/>

<b>5. COLLATERAL</b><br/>
Equipment purchased with loan proceeds serves as collateral<br/>
Lender holds first security interest in equipment<br/>
Borrower maintains insurance on collateral at full value<br/><br/>

<b>6. DEFAULT</b><br/>
Default triggers: Payment 15+ days late, bankruptcy filing, breach of covenants<br/>
Acceleration: Entire remaining balance becomes immediately due<br/>
Default Interest Rate: 12% per annum on unpaid balance<br/><br/>

Lender: ___________________  Date: _____________<br/>
Borrower: ___________________  Date: _____________
"""

create_contract(
    contracts_dir / "Loan_Agreement.pdf",
    "PROMISSORY NOTE AND LOAN AGREEMENT",
    loan_content,
)

# 4. SERVICE AGREEMENT
service_content = """
<b>SERVICE AGREEMENT</b><br/>
Date: January 12, 2025<br/><br/>

<b>SERVICE PROVIDER:</b> CloudTech Solutions, Inc., located at 888 Tech Boulevard, Seattle, WA 98101<br/>
<b>CLIENT:</b> MediCare Network, Inc., located at 444 Health Center Drive, Chicago, IL 60601<br/><br/>

<b>1. SCOPE OF SERVICES</b><br/>
Cloud Infrastructure Management: AWS infrastructure management and optimization<br/>
24/7 Monitoring: System uptime monitoring with 99.95% SLA guarantee<br/>
Technical Support: Premium technical support with 2-hour response time<br/>
Monthly Optimization: Performance tuning and cost optimization reports<br/><br/>

<b>2. TERM AND FEES</b><br/>
Initial Term: 12 months from March 1, 2025<br/>
Monthly Fee: USD 15,000 due on first of each month<br/>
Setup Fee: USD 5,000 (one-time)<br/>
Annual Increase: 3% per year for renewals<br/><br/>

<b>3. SERVICE LEVEL AGREEMENT</b><br/>
System Availability: 99.95% uptime guarantee<br/>
Response Time: Critical issues within 1 hour, standard within 4 hours<br/>
Monthly Maintenance Window: 4 hours during off-peak hours<br/>
Service Credits: 10% monthly credit for each hour below 99.95%<br/><br/>

<b>4. DELIVERABLES</b><br/>
Monthly Status Report: Performance metrics and recommendations<br/>
Security Audit: Quarterly security assessment reports<br/>
Capacity Planning: Quarterly resource forecasting reports<br/>
Documentation: Up-to-date system documentation and runbooks<br/><br/>

<b>5. PAYMENT TERMS</b><br/>
Due Date: Net 30 days from invoice<br/>
Payment Method: Bank transfer or ACH debit<br/>
Late Payment Fee: 1.5% per month on overdue amounts<br/><br/>

<b>6. TERMINATION</b><br/>
Either party may terminate with 60 days written notice<br/>
Early Termination Fee: 2 months of service fees if terminated before 12 months<br/>
Transition Support: 30 days transition support included<br/><br/>

Service Provider: ___________________  Date: _____________<br/>
Client: ___________________  Date: _____________
"""

create_contract(
    contracts_dir / "Service_Agreement.pdf", "SERVICE AGREEMENT", service_content
)

# 5. NDA (Non-Disclosure Agreement)
nda_content = """
<b>MUTUAL NON-DISCLOSURE AGREEMENT</b><br/>
Date: January 8, 2025<br/><br/>

<b>PARTY A:</b> InnovateTech Ventures, LLC, located at 333 Startup Way, San Francisco, CA 94105<br/>
<b>PARTY B:</b> FutureInvest Capital Partners, located at 777 Investment Drive, New York, NY 10005<br/><br/>

<b>1. PURPOSE</b><br/>
Parties are exploring potential business opportunity to develop and commercialize advanced AI technology platform.<br/>
Confidential Information may be disclosed to evaluate feasibility and potential partnership terms.<br/><br/>

<b>2. DEFINITION OF CONFIDENTIAL INFORMATION</b><br/>
Technical data, source code, algorithms, business plans, financial projections, customer lists, trade secrets<br/>
Proprietary methodologies, software, specifications, and know-how<br/>
Excludes: Public domain information, independently developed information, rightfully received from third parties<br/><br/>

<b>3. OBLIGATIONS</b><br/>
Receiving Party shall: Maintain strict confidentiality, protect information with reasonable security measures<br/>
Limit disclosure to employees/contractors with need-to-know on confidential basis<br/>
Use information only for stated evaluation purposes<br/>
Not reverse-engineer or attempt to derive underlying principles<br/><br/>

<b>4. DURATION</b><br/>
Confidentiality Period: 3 years from disclosure date<br/>
Survival: Obligations survive termination of discussions<br/>
Trade Secrets: Protected for as long as they remain trade secrets under applicable law<br/><br/>

<b>5. EXCEPTIONS</b><br/>
Receiving Party may disclose if required by law/court order<br/>
Prior written notice to Disclosing Party required when legally permitted<br/>
Allows cooperation with legal requirements while protecting confidentiality<br/><br/>

<b>6. REMEDIES</b><br/>
Breach causes irreparable harm for which monetary damages are inadequate<br/>
Disclosing Party entitled to injunctive relief and specific performance<br/>
Either party may pursue legal remedies for breach<br/><br/>

Party A: ___________________  Date: _____________<br/>
Party B: ___________________  Date: _____________
"""

create_contract(
    contracts_dir / "NDA_Agreement.pdf", "MUTUAL NON-DISCLOSURE AGREEMENT", nda_content
)

# 6. PARTNERSHIP AGREEMENT
partnership_content = """
<b>PARTNERSHIP AGREEMENT</b><br/>
Date: January 5, 2025<br/><br/>

<b>PARTNERS:</b><br/>
Partner A: Michael Chen, residing at 1234 Venture Street, San Francisco, CA 94103<br/>
Partner B: Jessica Williams, residing at 5678 Innovation Drive, San Francisco, CA 94104<br/>
Partner C: David Rodriguez, residing at 9012 Enterprise Lane, San Francisco, CA 94105<br/><br/>

<b>1. PARTNERSHIP NAME AND PURPOSE</b><br/>
Name: InnovateHub Technologies Partnership<br/>
Purpose: Develop and commercialize cloud-based business intelligence software<br/>
Principal Place of Business: 100 Tech Park, San Francisco, CA 94105<br/><br/>

<b>2. CAPITAL CONTRIBUTIONS</b><br/>
Partner A: USD 200,000 (40% ownership)<br/>
Partner B: USD 150,000 (30% ownership)<br/>
Partner C: USD 150,000 (30% ownership)<br/>
Total Capital: USD 500,000<br/><br/>

<b>3. PROFIT AND LOSS DISTRIBUTION</b><br/>
Partner A: 40% of net profits/losses<br/>
Partner B: 30% of net profits/losses<br/>
Partner C: 30% of net profits/losses<br/>
Distribution Schedule: Quarterly distributions of available cash flow<br/><br/>

<b>4. MANAGEMENT</b><br/>
Partner A: Chief Executive Officer and Chief Technology Officer<br/>
Partner B: Chief Financial Officer and Operations Manager<br/>
Partner C: Chief Product Officer and Business Development<br/>
Major Decisions: Require unanimous consent of all partners<br/>
Day-to-Day Operations: Individual partners may operate independently<br/><br/>

<b>5. CAPITAL CALLS</b><br/>
Additional capital may be required with 60 days notice<br/>
Each partner must contribute pro-rata to their ownership percentage<br/>
Failure to contribute results in dilution of ownership stake<br/><br/>

<b>6. WITHDRAWAL AND BUYOUT</b><br/>
Partner Withdrawal: 90 days written notice required<br/>
Valuation: Based on most recent partnership valuation<br/>
Buyout: Remaining partners have right of first refusal<br/>
Purchase Price: Fair market value determined by independent valuation<br/><br/>

<b>7. DISSOLUTION</b><br/>
Dissolution requires unanimous consent of all partners<br/>
Assets distributed pro-rata based on ownership percentages<br/>
Liabilities paid from partnership assets before distribution<br/><br/>

Partner A: ___________________  Date: _____________<br/>
Partner B: ___________________  Date: _____________<br/>
Partner C: ___________________  Date: _____________
"""

create_contract(
    contracts_dir / "Partnership_Agreement.pdf",
    "PARTNERSHIP AGREEMENT",
    partnership_content,
)

# 7. INVESTMENT AGREEMENT
investment_content = """
<b>SERIES A PREFERRED STOCK PURCHASE AGREEMENT</b><br/>
Date: January 1, 2025<br/><br/>

<b>COMPANY:</b> BlockChain Innovations Inc., a Delaware corporation<br/>
<b>INVESTOR:</b> Venture Capital Fund VII, LP, a Delaware limited partnership<br/><br/>

<b>1. INVESTMENT TERMS</b><br/>
Series A Preferred Stock Authorized: 5,000,000 shares<br/>
Purchase Price Per Share: USD 8.00<br/>
Total Investment Amount: USD 10,000,000<br/>
Number of Shares: 1,250,000 shares<br/><br/>

<b>2. INVESTOR RIGHTS</b><br/>
Board Seat: One designated board observer seat<br/>
Information Rights: Quarterly unaudited and annual audited financial statements<br/>
Pro-Rata Rights: Right to maintain ownership percentage in future financings<br/>
Drag-Along Rights: Must participate in company sale if approved by majority holders<br/><br/>

<b>3. VALUATION AND DILUTION</b><br/>
Pre-Money Valuation: USD 30,000,000<br/>
Post-Money Valuation: USD 40,000,000<br/>
Investor Ownership: 25% post-investment<br/>
Anti-Dilution: Weighted average anti-dilution protection in down rounds<br/><br/>

<b>4. LIQUIDATION PREFERENCES</b><br/>
Non-Cumulative Dividend: 8% annual return<br/>
Participating Preferred: Investors participate pro-rata in remaining proceeds<br/>
Preference Multiple: 1x non-participating in liquidation events<br/><br/>

<b>5. REDEMPTION RIGHTS</b><br/>
Redemption: If no qualified IPO by January 1, 2030, company must redeem shares<br/>
Redemption Price: Original purchase price plus accrued but unpaid dividends<br/>
Redemption Timeline: 12 months from redemption trigger event<br/><br/>

<b>6. USE OF PROCEEDS</b><br/>
Product Development: 40% (USD 4,000,000)<br/>
Sales and Marketing: 35% (USD 3,500,000)<br/>
Operations and Administration: 15% (USD 1,500,000)<br/>
Working Capital Reserve: 10% (USD 1,000,000)<br/><br/>

<b>7. REPRESENTATIONS AND WARRANTIES</b><br/>
Company: Duly organized, good standing, authorized to issue shares<br/>
Investor: Accredited investor, investment sophistication, not restricted person<br/>
No material undisclosed liabilities or legal proceedings<br/><br/>

Investor: ___________________  Date: _____________<br/>
Company CEO: ___________________  Date: _____________<br/>
Company Secretary: ___________________  Date: _____________
"""

create_contract(
    contracts_dir / "Investment_Agreement.pdf",
    "SERIES A PREFERRED STOCK PURCHASE AGREEMENT",
    investment_content,
)

print("\n✓ All contracts generated successfully!")
print(f"✓ Contracts saved to: {contracts_dir}")
