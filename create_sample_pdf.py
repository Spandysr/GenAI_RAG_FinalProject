"""
create_sample_pdf.py — Generate a sample customer support PDF for testing
==========================================================================
Creates a realistic customer support FAQ document that demonstrates
the ingestion pipeline and provides content for retrieval testing.
"""

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import textwrap
import os


def create_sample_pdf(output_path="data/sample_support_docs.pdf"):
    """Generate a multi-topic customer support FAQ PDF."""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    y = height - 50

    def write_line(text, font="Helvetica", size=11, bold=False):
        nonlocal y
        if bold:
            font = "Helvetica-Bold"
        c.setFont(font, size)
        # Wrap long lines
        lines = textwrap.wrap(text, width=85)
        for line in lines:
            if y < 50:
                c.showPage()
                y = height - 50
                c.setFont(font, size)
            c.drawString(50, y, line)
            y -= size + 4

    def write_blank():
        nonlocal y
        y -= 15

    # Title
    write_line("ACME Corp — Customer Support Knowledge Base", size=16, bold=True)
    write_line("Last Updated: April 2026", size=10)
    write_blank()
    write_blank()

    # Section 1: Returns
    write_line("1. RETURN POLICY", size=14, bold=True)
    write_blank()
    write_line("Our return policy allows customers to return most items within 30 days of purchase.")
    write_line("Items must be in their original packaging and in unused condition.")
    write_line("To initiate a return, log in to your account, go to Order History, and click 'Return Item'.")
    write_line("Refunds are processed within 5-7 business days after we receive the returned item.")
    write_line("Shipping costs for returns are covered by ACME Corp for defective items.")
    write_line("For non-defective returns, a flat shipping fee of $5.99 is deducted from the refund.")
    write_line("Electronics and perishable items cannot be returned after opening.")
    write_blank()

    # Section 2: Account
    write_line("2. ACCOUNT MANAGEMENT", size=14, bold=True)
    write_blank()
    write_line("To reset your password, click 'Forgot Password' on the login page.")
    write_line("Enter your registered email address and you will receive a reset link within 5 minutes.")
    write_line("If you do not receive the email, check your spam folder or contact support.")
    write_line("To update your email address, go to Settings > Account > Email and verify the new address.")
    write_line("Two-factor authentication (2FA) can be enabled from Settings > Security.")
    write_line("We recommend enabling 2FA for enhanced account security.")
    write_line("To delete your account, contact our support team. Account deletion is permanent and")
    write_line("all data including order history will be removed within 30 days.")
    write_blank()

    # Section 3: Shipping
    write_line("3. SHIPPING INFORMATION", size=14, bold=True)
    write_blank()
    write_line("Standard shipping takes 5-7 business days and is free for orders over $50.")
    write_line("Express shipping (2-3 business days) is available for $9.99.")
    write_line("Overnight shipping is available for $19.99 for orders placed before 2 PM EST.")
    write_line("International shipping is available to select countries and takes 10-15 business days.")
    write_line("All orders include tracking information sent via email once shipped.")
    write_line("If your order has not arrived within the estimated delivery window, please contact support.")
    write_blank()

    # Section 4: Contact
    write_line("4. CONTACT INFORMATION & BUSINESS HOURS", size=14, bold=True)
    write_blank()
    write_line("Our customer support team is available Monday through Friday, 9 AM to 6 PM EST.")
    write_line("Saturday support hours are 10 AM to 4 PM EST. We are closed on Sundays.")
    write_line("You can reach us via email at support@acmecorp.com or call 1-800-ACME-HELP.")
    write_line("Live chat is available on our website during business hours.")
    write_line("For urgent issues outside business hours, email urgent@acmecorp.com.")
    write_line("Average response time for email inquiries is 4-6 hours during business days.")
    write_blank()

    # Section 5: Products
    write_line("5. PRODUCT WARRANTY", size=14, bold=True)
    write_blank()
    write_line("All ACME products come with a standard 1-year manufacturer warranty.")
    write_line("The warranty covers defects in materials and workmanship under normal use.")
    write_line("Extended warranty plans (2-year, 3-year) can be purchased at checkout.")
    write_line("To file a warranty claim, provide your order number and photos of the defect.")
    write_line("Warranty claims are reviewed within 3-5 business days.")
    write_line("Approved claims result in a replacement product shipped at no additional cost.")
    write_blank()

    # Section 6: Pricing & Billing
    write_line("6. PRICING AND BILLING", size=14, bold=True)
    write_blank()
    write_line("We accept Visa, MasterCard, American Express, PayPal, and Apple Pay.")
    write_line("All prices are listed in USD and include applicable taxes at checkout.")
    write_line("Promotional codes can be applied at checkout in the 'Promo Code' field.")
    write_line("Only one promotional code can be used per order.")
    write_line("For billing disputes, please contact our billing department directly.")
    write_line("Billing disputes must be raised within 60 days of the transaction date.")
    write_line("We do not store complete credit card numbers — only the last 4 digits for reference.")
    write_blank()

    # Section 7: Subscription
    write_line("7. SUBSCRIPTION SERVICES", size=14, bold=True)
    write_blank()
    write_line("ACME Plus subscription is $14.99/month or $149.99/year (save 17%).")
    write_line("Benefits include free express shipping, early access to sales, and priority support.")
    write_line("You can cancel your subscription at any time from Settings > Subscription.")
    write_line("Cancellation takes effect at the end of the current billing period.")
    write_line("No refunds are provided for partial billing periods.")

    c.save()
    print(f"[OK] Sample PDF created: {output_path}")
    return output_path


if __name__ == "__main__":
    create_sample_pdf()
