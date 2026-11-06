"""Données supplémentaires : emails transactionnels synthétiques + bruit d'augmentation.

Pourquoi ? Dans le dataset d'origine, presque tous les emails « une entreprise écrit à son
client » (banque, livraison, abonnement...) sont du phishing. Le modèle a donc appris :
« banque / transaction / your account » => phishing, et classe à tort les vraies notifications.

Principe : pour chaque scénario (transaction bancaire, colis, mot de passe...), on génère
une version LÉGITIME et une version PHISHING qui partagent les mêmes marques, salutations,
signatures et bas de page. Seuls les vrais indices diffèrent :
    - domaine du lien (officiel  vs  sosie, adresse IP, lien raccourci)
    - demande d'informations (aucune  vs  mot de passe, carte, code, virement)
    - pression (aucune  vs  menace de blocage, délai court)
    - canal de contact (appli / numéro au dos de la carte  vs  « cliquez ici », « répondez »)
Ainsi le modèle ne peut pas apprendre « banque = légitime » : il doit regarder les indices.

Généralisation : une partie des marques et des formulations (1 sur 5) est réservée au jeu
de test synthétique « test_synth », jamais vue pendant l'entraînement.

Bruit (train uniquement, mêmes probabilités pour les 2 classes) : fautes de frappe,
caractères déguisés (0 pour o), mots supprimés, phrases mélangées, troncature, réponses
et transferts, signatures mobiles, avertissements légaux, résidus HTML.
La casse et les espaces ne sont pas bruités : normalize() les efface de toute façon.
"""
import random
import re

# --------------------------------------------------------------------------- marques
BRANDS = {
    "bank": [("Chase", "chase.com"), ("Wells Fargo", "wellsfargo.com"), ("Barclays", "barclays.co.uk"),
             ("HSBC", "hsbc.com"), ("Citibank", "citi.com"), ("Capital One", "capitalone.com"),
             ("Bank of America", "bankofamerica.com"), ("Santander", "santander.com"),
             ("TD Bank", "td.com"), ("Ally Bank", "ally.com"), ("US Bank", "usbank.com"),
             ("PNC Bank", "pnc.com"), ("Discover", "discover.com"), ("Lloyds Bank", "lloydsbank.com"),
             ("Revolut", "revolut.com")],
    "shop": [("Amazon", "amazon.com"), ("eBay", "ebay.com"), ("Walmart", "walmart.com"),
             ("Target", "target.com"), ("Best Buy", "bestbuy.com"), ("Etsy", "etsy.com"),
             ("IKEA", "ikea.com"), ("Zalando", "zalando.com"), ("Home Depot", "homedepot.com"),
             ("Costco", "costco.com")],
    "carrier": [("UPS", "ups.com"), ("FedEx", "fedex.com"), ("DHL", "dhl.com"), ("USPS", "usps.com"),
                ("Royal Mail", "royalmail.com"), ("Canada Post", "canadapost.ca"), ("Purolator", "purolator.com"),
                ("Evri", "evri.com"), ("GLS", "gls-group.com"), ("Aramex", "aramex.com")],
    "service": [("Netflix", "netflix.com"), ("Spotify", "spotify.com"), ("Microsoft", "microsoft.com"),
                ("Google", "google.com"), ("Apple", "apple.com"), ("Dropbox", "dropbox.com"),
                ("Adobe", "adobe.com"), ("LinkedIn", "linkedin.com"), ("Zoom", "zoom.us"),
                ("Slack", "slack.com"), ("Disney+", "disneyplus.com"), ("Hulu", "hulu.com"),
                ("Canva", "canva.com"), ("GitHub", "github.com"), ("Uber", "uber.com")],
    "payroll": [("ADP", "adp.com"), ("Workday", "workday.com"), ("Gusto", "gusto.com"),
                ("Paychex", "paychex.com"), ("BambooHR", "bamboohr.com")],
    "gov": [("IRS", "irs.gov"), ("HMRC", "gov.uk"), ("Social Security Administration", "ssa.gov"),
            ("DMV", "dmv.ca.gov"), ("Canada Revenue Agency", "canada.ca")],
}


def held_out(i: int) -> bool:
    """1 élément sur 5 est réservé au test synthétique."""
    return i % 5 == 4


def pick(rng, pool, split):
    items = [x for i, x in enumerate(pool) if held_out(i) == (split == "test")]
    return rng.choice(items or pool)


FIRST = ["James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda", "David",
         "Elizabeth", "William", "Susan", "Daniel", "Sarah", "Thomas", "Karen", "Kevin", "Emily",
         "Brian", "Laura", "Ahmed", "Mei", "Carlos", "Fatima", "Olivia"]
LAST = ["Smith", "Johnson", "Brown", "Garcia", "Miller", "Davis", "Wilson", "Taylor", "Clark",
        "Lewis", "Walker", "Young", "King", "Wright", "Lopez", "Hill", "Green", "Baker", "Nguyen", "Patel"]
CITIES = ["Chicago", "Houston", "Phoenix", "London", "Toronto", "Madrid", "Lagos", "Mumbai",
          "Sao Paulo", "Moscow", "Manila", "Dallas", "Berlin", "Warsaw", "Kyiv"]
DEVICES = ["Chrome on Windows", "Safari on iPhone", "Firefox on Linux", "Edge on Windows",
           "Chrome on Android", "the Mobile App on iPad", "Safari on Mac"]
ITEMS = ["wireless headphones", "a coffee maker", "running shoes", "a phone case", "a desk lamp",
         "a backpack", "a bluetooth speaker", "kitchen knives", "a yoga mat", "a winter jacket",
         "an office chair", "a smart watch"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]

# --------------------------------------------------------------------------- parties communes
GREETINGS = ["Dear Customer,", "Dear {first},", "Hello {first},", "Hi {first},", "Dear {first} {last},",
             "Dear Valued Customer,", "Hello,", "Dear Member,", "Hi there,", "Dear Account Holder,"]
CLOSINGS = ["Thank you for choosing {brand}.", "Thank you for being a {brand} customer.",
            "Thanks for using {brand}.", "We appreciate your business.", "Thank you.",
            "Thank you for banking with us.", "We are here to help.", "Have a great day.",
            "Thank you for your trust.", "Kind regards and thank you."]
SIGNATURES = ["Best regards,\n{brand} Customer Support", "Sincerely,\nThe {brand} Team",
              "Regards,\n{brand} Account Services", "The {brand} Team", "{brand} Customer Care",
              "Best,\n{brand} Support Team", "Kind regards,\n{brand} Security Team",
              "Thanks,\n{brand} Notifications", "Warm regards,\n{brand} Client Services",
              "{brand} Online Services"]
FOOTERS = ["", "", "",
           "This is an automated message, please do not reply to this email.",
           "You are receiving this email because you have an account with {brand}.",
           "{brand}, Inc. All rights reserved. Privacy Policy | Terms of Use",
           "To manage your email preferences, visit your account settings.",
           "Please do not reply to this message. Replies to this address are not monitored.",
           "{brand} will never ask you for your password or full card number by email.",
           "Copyright {year} {brand}. All rights reserved."]
SUBJECTS = {
    "transaction": ["Transaction alert", "Your recent transaction", "Card transaction notice",
                    "Payment processed", "Activity on your account"],
    "statement": ["Your statement is ready", "New statement available", "Your monthly statement",
                  "Account statement notification", "Your e-statement"],
    "signin": ["New sign-in to your account", "Security notice", "Sign-in alert",
               "New device detected", "Account activity notice"],
    "password": ["Your password was changed", "Password update confirmation", "Security update",
                 "Account password changed", "Your credentials were updated"],
    "order": ["Your order confirmation", "Order received", "Thanks for your order",
              "Order #{order} confirmed", "We received your order"],
    "shipping": ["Your package is on the way", "Delivery update", "Shipment notification",
                 "Your parcel", "Tracking update for your shipment"],
    "subscription": ["Your subscription", "Membership renewal", "Billing update",
                     "Your plan renews soon", "Payment information"],
    "otp": ["Your verification code", "Security code", "One-time passcode",
            "Your sign-in code", "Verification required"],
    "payroll": ["Your pay statement", "Payroll notification", "Direct deposit update",
                "Your payslip", "Pay stub available"],
    "storage": ["Storage almost full", "Your storage", "Mailbox quota", "Account storage notice",
                "Storage usage alert"],
    "tax": ["Tax notice", "Your tax return", "Refund status", "Tax refund information",
            "Notice from {brand}"],
    "shared": ["Document shared with you", "{first2} shared a file", "New shared document",
               "File shared with you", "You have a new document"],
}

# --------------------------------------------------------------------------- scénarios
# Pour chaque scénario : secteur de la marque, corps commun, action légitime, action phishing
SCENARIOS = {
    "transaction": dict(
        sector="bank",
        body=["Your recent transaction of {amount} was successfully processed on {date}.",
              "A purchase of {amount} was made with your card ending in {last4} on {date}.",
              "We processed a payment of {amount} from your account on {date}.",
              "Your card ending in {last4} was used for {amount} at an online merchant on {date}.",
              "A transfer of {amount} was completed from your checking account on {date}."],
        legit=["You can review the transaction details in your online banking application.",
               "If you do not recognize this transaction, please call the number on the back of your card.",
               "No action is needed if you made this purchase. You can see all activity in the {brand} app.",
               "If something looks wrong, sign in to the {brand} app or visit a branch.",
               "You can review recent activity anytime at {url_legit}."],
        phish=["If you do not recognize this transaction, cancel it immediately at {url_phish}.",
               "If this was not you, verify your identity within 24 hours at {url_phish} or your card will be blocked.",
               "To dispute this charge, reply with your full card number, expiry date and CVV.",
               "Your account has been temporarily limited. Restore access at {url_phish}.",
               "To stop this payment, confirm your online banking login here: {url_phish}"]),
    "statement": dict(
        sector="bank",
        body=["Your {month} statement is now available.",
              "Your account statement for {month} is ready to view.",
              "A new e-statement has been added to your account.",
              "Your monthly statement for the account ending in {last4} is available.",
              "Your credit card statement for {month} has been generated."],
        legit=["You can view it by signing in to online banking as usual.",
               "To view it, open the {brand} app and go to Statements.",
               "Sign in at {url_legit} to view and download your statement.",
               "Your statement is available in the Documents section of your account.",
               "There is nothing you need to do right now."],
        phish=["Due to a security update, you must re-confirm your details at {url_phish} to view it.",
               "Download the attached statement and enter your login details to open it.",
               "Your statement could not be delivered. Update your information at {url_phish} within 48 hours.",
               "View your statement here: {url_phish} (login required to avoid account suspension).",
               "Unusual charges were found. Verify your account now at {url_phish}."]),
    "signin": dict(
        sector="service",
        body=["We noticed a new sign-in to your {brand} account from {device} near {city}.",
              "Your {brand} account was accessed from a new device ({device}) on {date}.",
              "A new sign-in was detected on your account on {date}.",
              "Someone signed in to your {brand} account from {city}.",
              "We detected a login from a device we did not recognize: {device}."],
        legit=["If this was you, you can ignore this email.",
               "If this was not you, open your account settings and change your password.",
               "You can review your recent activity at {url_legit}.",
               "If you do not recognize this activity, go to Security in the {brand} app.",
               "No further action is required if this was you."],
        phish=["If this was not you, secure your account immediately at {url_phish}.",
               "Your account will be suspended in 24 hours unless you verify your identity at {url_phish}.",
               "To keep your account, confirm your password here: {url_phish}",
               "Reply to this email with your username and password so we can secure your account.",
               "Click here to lock your account now: {url_phish}"]),
    "password": dict(
        sector="service",
        body=["The password for your {brand} account was changed on {date}.",
              "Your {brand} password was successfully updated.",
              "This email confirms that your account password was reset on {date}.",
              "You recently changed the password for your {brand} account.",
              "Your account credentials were updated on {date}."],
        legit=["If you made this change, no further action is needed.",
               "If you did not make this change, please contact support through the {brand} website.",
               "If this was not you, reset your password from the {brand} app.",
               "You can review your security settings at {url_legit}.",
               "If you did not request this, visit {url_legit} to secure your account."],
        phish=["If you did not make this change, cancel the request here: {url_phish}",
               "If this was not you, log in at {url_phish} within 12 hours to keep your account.",
               "To undo this change, send us your old password and your phone number.",
               "Your account will be deleted unless you confirm your identity at {url_phish}.",
               "If you did not request this, verify your billing information at {url_phish}."]),
    "order": dict(
        sector="shop",
        body=["Thank you for your order #{order} of {item} for {amount}.",
              "We received your order #{order}. Total: {amount}.",
              "Your order of {item} has been confirmed.",
              "Order #{order} ({item}) was placed on {date}.",
              "Thanks for shopping with {brand}! Your order #{order} is being prepared."],
        legit=["You can track your order in Your Orders at {url_legit}.",
               "We will send you another email when your items ship.",
               "You can view or manage your order in the {brand} app.",
               "Estimated delivery: {date2}.",
               "Questions? Visit the Help section of our website."],
        phish=["There was a problem with your payment. Update your card details at {url_phish} or your order will be cancelled.",
               "If you did not place this order, cancel it now at {url_phish}.",
               "Your order is on hold. Confirm your billing address and card number at {url_phish}.",
               "To receive a refund, enter your bank login details here: {url_phish}",
               "We could not verify your identity. Click {url_phish} within 24 hours."]),
    "shipping": dict(
        sector="carrier",
        body=["Your package with tracking number {track} is on its way.",
              "Your shipment {track} is scheduled for delivery on {date2}.",
              "We attempted to deliver your package {track} on {date}.",
              "Your parcel from {shop} has been shipped.",
              "Your package {track} is out for delivery today."],
        legit=["You can track it at {url_legit}.",
               "No action is required. Your package will be delivered as planned.",
               "You can change the delivery date in the {brand} app.",
               "We will try again on the next business day.",
               "Track your package on our website using the tracking number above."],
        phish=["A redelivery fee of $1.99 is required. Pay now at {url_phish}.",
               "Please confirm your address within 48 hours at {url_phish} or your package will be returned.",
               "Your package is held at customs. Pay the duty fee here: {url_phish}",
               "Delivery failed. Update your payment details at {url_phish} to reschedule.",
               "Download the attached label and enter your card details to release your parcel."]),
    "subscription": dict(
        sector="service",
        body=["Your {brand} {plan} plan will renew on {date2} for {amount}.",
              "Thank you for your payment of {amount} for your {brand} membership.",
              "We were unable to process your payment of {amount} for your {plan} plan.",
              "Your {brand} subscription has been renewed.",
              "Your free trial of {brand} {plan} ends on {date2}."],
        legit=["You can manage or cancel your plan anytime in your account settings.",
               "To update your payment method, go to Account > Billing in the {brand} app.",
               "We will try again in a few days. You can update your card at {url_legit}.",
               "Your receipt is available in your account.",
               "No action is needed to continue your membership."],
        phish=["Your account will be suspended today unless you update your billing information at {url_phish}.",
               "Confirm your credit card details within 24 hours at {url_phish} to avoid losing access.",
               "Reply with your card number and security code to keep your subscription.",
               "Your membership has been suspended. Reactivate it here: {url_phish}",
               "To get a refund of {amount}, sign in with your bank details at {url_phish}."]),
    "otp": dict(
        sector="service",
        body=["Your {brand} verification code is {code}.",
              "Use code {code} to sign in to your {brand} account.",
              "Here is your one-time passcode: {code}.",
              "Your security code is {code}. It expires in 10 minutes.",
              "Enter {code} to verify your email address."],
        legit=["Never share this code with anyone. {brand} will never ask you for it.",
               "If you did not request this code, you can ignore this email.",
               "This code can only be used once.",
               "Do not share this code. Our staff will never call you to ask for it.",
               "If you did not try to sign in, change your password in the app."],
        phish=["Reply to this email with the code to confirm your account.",
               "A support agent will call you. Please give them this code to verify your identity.",
               "If you did not request this code, enter it at {url_phish} to cancel the request.",
               "To avoid suspension, confirm the code and your password at {url_phish}.",
               "Forward this code to our security team at {fake_email} immediately."]),
    "payroll": dict(
        sector="payroll",
        body=["Your pay statement for {month} is now available.",
              "Your payslip for the period ending {date} has been posted.",
              "A change was requested to your direct deposit information.",
              "Your W-2 form for {year} is ready.",
              "Payroll for {month} has been processed."],
        legit=["Sign in to the {brand} portal as usual to view it.",
               "You can view it in the employee portal or the {brand} mobile app.",
               "If you did not request this change, contact your HR department directly.",
               "Questions about your pay? Contact your payroll administrator.",
               "No action is required."],
        phish=["Verify your direct deposit details at {url_phish} or your salary will be delayed.",
               "Confirm your bank account and login details here to receive your payment: {url_phish}",
               "Your payment is on hold. Update your banking information within 24 hours at {url_phish}.",
               "Open the attached file and enable macros to view your payslip.",
               "Reply with your bank account number and routing number to update your deposit."]),
    "storage": dict(
        sector="service",
        body=["You have used 90% of your {brand} storage.",
              "Your mailbox is almost full.",
              "Your {brand} storage is full. New files will not sync.",
              "You are running out of space in your {brand} account.",
              "Your account has reached its storage limit."],
        legit=["You can free up space or upgrade your plan in your account settings.",
               "Visit {url_legit} to manage your storage.",
               "Delete old files or upgrade from the {brand} app.",
               "Learn more about storage plans on our website.",
               "No action is required if you do not need more space."],
        phish=["Click here to increase your quota for free: {url_phish}",
               "Verify your account within 24 hours at {url_phish} or your emails will be deleted.",
               "Sign in at {url_phish} to restore your mailbox.",
               "Your account will be deactivated. Confirm your password here: {url_phish}",
               "Reply with your email password to receive additional storage."]),
    "tax": dict(
        sector="gov",
        body=["We have received your {year} tax return.",
              "Your tax refund of {amount} has been processed.",
              "Your tax return for {year} is being reviewed.",
              "A notice about your {year} tax account has been issued.",
              "You are eligible for a tax refund of {amount}."],
        legit=["You can check the status of your refund in your online account on the official website.",
               "You will receive a letter by mail with more details.",
               "No action is required at this time.",
               "Check your refund status at {url_legit}.",
               "If you have questions, call the number listed on your official notice."],
        phish=["To receive your refund, submit your bank card details at {url_phish}.",
               "Claim your refund within 48 hours at {url_phish} or it will be cancelled.",
               "Reply with your Social Security Number and bank account to receive payment.",
               "Failure to verify your identity at {url_phish} will result in legal action.",
               "Download the attached form, fill in your card details and send it back."]),
    "shared": dict(
        sector="service",
        body=["{first2} {last2} shared a document with you: \"{doc}\".",
              "{first2} has invited you to view \"{doc}\".",
              "A new file \"{doc}\" was shared with you.",
              "{first2} {last2} shared the folder \"{doc}\" with you.",
              "You have been given access to \"{doc}\"."],
        legit=["Open it from the {brand} app or at {url_legit}.",
               "You can find it in Shared with me.",
               "Open the document in {brand}.",
               "You received this email because {first2} shared a file with you.",
               "View it at {url_legit}."],
        phish=["Sign in with your email password to view the document: {url_phish}",
               "This document will expire in 24 hours. Open it now at {url_phish}.",
               "Verify your email account to access the file at {url_phish}.",
               "Enter your Microsoft 365 credentials at {url_phish} to open the file.",
               "Download the attached HTML file and log in to view the document."]),
}
DOCS = ["Q3 budget", "Invoice 2291", "Team schedule", "Contract draft", "Salary review",
        "Project plan", "Meeting notes", "Payment details", "HR policy update", "Shared photos"]
PLANS = ["Premium", "Standard", "Family", "Basic", "Pro", "Plus", "Business", "Student"]
URGENCY = ["URGENT: ", "Action required: ", "Final notice: ", "Important: ", "Immediate attention: "]


def phish_url(rng, name, domain, split):
    host = domain.split(".")[0]
    words = ["secure", "login", "verify", "account", "update", "support", "billing", "alert", "service"]
    w1, w2 = rng.sample(words, 2)
    tld = rng.choice(["com", "net", "info", "online", "xyz", "co", "top", "site"])
    leet = host.replace("o", "0").replace("l", "1").replace("i", "1").replace("e", "3")
    patterns = [
        f"http://{host}-{w1}-{w2}.{tld}/login",
        f"http://{w1}-{host}.{tld}/{w2}",
        f"http://{rng.randint(11, 223)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}/{host}/{w1}",
        f"http://{host}.{w1}-{w2}.{tld}/",
        f"https://{rng.choice(['bit.ly', 'tinyurl.com', 'is.gd', 'cutt.ly', 'rb.gy'])}/{rng.randint(10000, 99999)}x",
        f"http://www.{leet}.{tld}/{w1}",
        f"https://{host}{rng.randint(1, 99)}.{tld}/{w1}/{w2}",
        f"http://{domain}.{w1}.{tld}/session",
        f"https://{w1}.{host}-{tld}.com/",
        f"http://{host}-{w2}.web.app/{w1}",
    ]
    return pick(rng, patterns, split)


def legit_url(rng, domain, split):
    paths = ["", "account", "help", "security", "activity", "orders", "settings", "billing",
             "myaccount", "support"]
    p = pick(rng, paths, split)
    return f"https://www.{domain}/{p}" if p else f"https://www.{domain}"


def make_email(rng, label: int, split: str) -> str:
    kind = rng.choice(list(SCENARIOS))
    sc = SCENARIOS[kind]
    name, domain = pick(rng, BRANDS[sc["sector"]], split)
    shop = pick(rng, BRANDS["shop"], split)[0]
    day = rng.randint(1, 28)
    ctx = dict(
        brand=name, shop=shop, first=rng.choice(FIRST), last=rng.choice(LAST),
        first2=rng.choice(FIRST), last2=rng.choice(LAST),
        amount=rng.choice(["$", "$", "£", "€"]) + f"{rng.uniform(4, 2500):,.2f}",
        date=f"{rng.choice(MONTHS)} {day}, {rng.randint(2021, 2026)}",
        date2=f"{rng.choice(MONTHS)} {rng.randint(1, 28)}", month=rng.choice(MONTHS),
        year=rng.randint(2020, 2026), last4=f"{rng.randint(0, 9999):04d}",
        order=f"{rng.randint(100, 999)}-{rng.randint(1000000, 9999999)}",
        track=rng.choice(["1Z", "", "JD", "LX"]) + f"{rng.randint(10**9, 10**10 - 1)}",
        item=rng.choice(ITEMS), city=rng.choice(CITIES), device=rng.choice(DEVICES),
        code=f"{rng.randint(0, 999999):06d}", plan=rng.choice(PLANS), doc=rng.choice(DOCS),
        url_legit=legit_url(rng, domain, split), url_phish=phish_url(rng, name, domain, split),
        fake_email=f"{rng.choice(['security', 'verify', 'support'])}@{domain.split('.')[0]}-help.{rng.choice(['com', 'net'])}",
    )
    action_pool = sc["legit"] if label == 0 else sc["phish"]
    parts = []
    if rng.random() < 0.5:
        subj = pick(rng, SUBJECTS[kind], split)
        if label == 1 and rng.random() < 0.4:
            subj = rng.choice(URGENCY) + subj
        parts.append("Subject: " + subj)
    parts.append(pick(rng, GREETINGS, split))
    parts.append(pick(rng, sc["body"], split))
    parts.append(pick(rng, action_pool, split))
    # Une 2e action (même classe) dans 40 % des cas : emails plus variés
    if rng.random() < 0.4:
        extra = rng.choice(action_pool)
        if extra != parts[-1]:
            parts.append(extra)
    parts.append(pick(rng, CLOSINGS, split))
    parts.append(pick(rng, SIGNATURES, split))
    footer = rng.choice(FOOTERS)       # mêmes bas de page pour les 2 classes (même la phrase
    if footer:                         # « nous ne demandons jamais votre mot de passe »)
        parts.append(footer)
    text = "\n\n".join(parts)
    return text.format(**ctx)


def synthetic(n_legit: int, n_phish: int, split: str, seed: int):
    rng = random.Random(seed)
    rows = [(make_email(rng, 0, split), 0) for _ in range(n_legit)]
    rows += [(make_email(rng, 1, split), 1) for _ in range(n_phish)]
    rng.shuffle(rows)
    return rows


# =========================================================================== bruit
_HOMO = {"o": "0", "l": "1", "i": "1", "e": "3", "a": "@", "s": "5"}
_KEYS = "abcdefghijklmnopqrstuvwxyz"

PREFIXES = ["fw :", "fwd :", "re :", "re : re :", "subject : fwd :"]
QUOTES = [
    "- - - - - original message - - - - - from : {who} sent : monday , {date} to : {who2} subject : re : update",
    "on {date} , {who} wrote :",
    "- - - - - forwarded by {who} on {date} - - - - -",
    "begin forwarded message : from : {who} date : {date}",
]
SUFFIXES = [
    "sent from my iphone", "sent from my android device", "get outlook for ios",
    "this email and any attachments are confidential and intended solely for the addressee . "
    "if you have received this email in error please notify the sender .",
    "please consider the environment before printing this email .",
    "to unsubscribe from this list , send an email with \" unsubscribe \" in the subject line .",
    "< br > < br > & nbsp ;", "< / div > < / body > < / html >",
    "this message was scanned for viruses .",
    "view this email in your browser",
]


def _typo(word, rng):
    if len(word) < 4 or not word.isalpha():
        return word
    i = rng.randrange(len(word) - 1)
    op = rng.randrange(4)
    if op == 0:
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]     # inversion
    if op == 1:
        return word[:i] + word[i + 1:]                              # lettre manquante
    if op == 2:
        return word[:i] + word[i] + word[i:]                        # lettre doublée
    return word[:i] + rng.choice(_KEYS) + word[i + 1:]              # mauvaise touche


def add_noise(text: str, rng: random.Random) -> str:
    """Applique 1 à 3 perturbations à un texte DÉJÀ normalisé (minuscules, ponctuation isolée)."""
    words = text.split()
    ops = rng.sample(["typo", "homoglyph", "drop", "shuffle", "truncate", "prefix", "quote", "suffix"],
                     k=rng.randint(1, 3))
    for op in ops:
        if op == "typo":
            words = [_typo(w, rng) if rng.random() < 0.05 else w for w in words]
        elif op == "homoglyph":
            words = ["".join(_HOMO.get(c, c) if rng.random() < 0.3 else c for c in w)
                     if w.isalpha() and len(w) > 3 and rng.random() < 0.04 else w for w in words]
        elif op == "drop" and len(words) > 20:
            words = [w for w in words if rng.random() > 0.05]
        elif op == "shuffle":
            joined = " ".join(words)
            # Après normalize(), « site.com » devient « site . com » : couper aux points
            # casserait les liens en morceaux. On ne mélange que les emails sans lien.
            if re.search(r"http|www|\. (com|net|org|co|uk|gov|io|us)\b", joined):
                continue
            sents = re.split(r"(?<= [.!?]) ", joined)
            if len(sents) > 2:
                head, rest = sents[0], sents[1:]
                rng.shuffle(rest)
                words = " ".join([head] + rest).split()
        elif op == "truncate" and len(words) > 40:
            words = words[: int(len(words) * rng.uniform(0.6, 0.9))]
        elif op == "prefix":
            words = rng.choice(PREFIXES).split() + words
        elif op == "quote":
            q = rng.choice(QUOTES).format(
                who=f"{rng.choice(FIRST)} {rng.choice(LAST)}".lower(),
                who2=f"{rng.choice(FIRST)} {rng.choice(LAST)}".lower(),
                date=f"{rng.choice(MONTHS)} {rng.randint(1, 28)} , {rng.randint(2001, 2026)}".lower())
            words = (q.split() + words) if rng.random() < 0.5 else (words + q.split())
        elif op == "suffix":
            words = words + rng.choice(SUFFIXES).split()
    return " ".join(words)
