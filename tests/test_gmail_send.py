import base64
import os.path

from email.message import EmailMessage

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


SCOPES = [
    "https://www.googleapis.com/auth/gmail.send"
]


def get_credentials():
    creds = None

    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file(
            "token.json",
            SCOPES,
        )

    if not creds or not creds.valid:

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())

        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json",
                SCOPES,
            )

            creds = flow.run_local_server(
                port=0
            )

        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return creds


def send_email():
    creds = get_credentials()

    service = build(
        "gmail",
        "v1",
        credentials=creds,
    )

    message = EmailMessage()

    message.set_content(
        "Hello from my RAG Agent!"
    )

    message["To"] = "alichreim875@gmail.com"

    message["Subject"] = "RAG Agent Test"

    encoded_message = base64.urlsafe_b64encode(
        message.as_bytes()
    ).decode()

    request_body = {
        "raw": encoded_message
    }

    result = (
        service.users()
        .messages()
        .send(
            userId="me",
            body=request_body,
        )
        .execute()
    )

    print(
        "Email sent successfully!"
    )

    print(
        "Message ID:",
        result["id"],
    )


if __name__ == "__main__":
    send_email()