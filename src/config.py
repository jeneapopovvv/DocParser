


supported_documents = {
    "passport":         "Passport",
    "identity_card":    "Government ID",
    "driver_license":   "Driver license",
    "iban":             "Financial account statements from banking institutions",
    "lease_agreement":  "Lease agreement contract between landlord and tenant",
    "cr_certificate":   "Commercial Registration Certificate issued by the government",
}

document_schemas = {
    "passport": {
        "passportNumber": "string",
        "nationality": "string",
        "fullName": "string",
        "fullNameArabic": "string",
        "dateOfBirth": "date",
        "expiryDate": "date",
        "gender": "string"
    },
    "identity_card": {
        "personalNumber": "number",
        "nationality": "string",
        "fullName": "string",
        "fullNameArabic": "string",
        "dateOfBirth": "date",
        "expiryDate": "date",
        "gender": "string"
    },
    "driver_license": {
        "licenseNumber": "number",
        "nationality": "string",
        "fullName": "string",
        "fullNameArabic": "string",
        "dateOfBirth": "date",
        "gender": "string"
    },
    "iban": {
        "fullName": "string",
        "bankAccountNumber": "string",
        "iban": "string",
        "branch": "string",
        "date": "date",
        "bic/swift": "string"
    },
    "lease_agreement": {
        "landlords": [{
            "name": "string",
            "email": "string",
            "phone": "string",
            "cpr": "string"
        }],
        "tenants": [{
            "name": "string",
            "email": "string",
            "phone": "string",
            "cpr": "string"
        }],
        "propertyAddress": "string",
        "propertyType": "string",
        "leaseStartDate": "date",
        "leaseEndDate": "date"
    },
    "cr_certificate": {
        "registrationNumber": "string",
        "commercialName": "string",
        "commercialNameArabic": "string",
        "type": "string",   
        "status": "string",
        "issueDate": "date",
        "expiryDate": "date",
    }
}

