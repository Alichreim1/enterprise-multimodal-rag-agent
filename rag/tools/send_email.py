import base64
import os

from email.message import EmailMessage

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from rag.tools.base_tool import BaseTool


SCOPES = [
    "https://www.googleapis.com/auth/gmail.send"
]


class SendEmailTool(BaseTool):

    @property
    def name(self) -> str:
        return "send_email"

    @property
    def description(self) -> str:
        return "Send an email using Gmail."

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {
                            "type": "string",
                            "description": (
                                "Recipient email address."
                            ),
                        },
                        "subject": {
                            "type": "string",
                            "description": (
                                "Email subject."
                            ),
                        },
                        "body": {
                            "type": "string",
                            "description": (
                                "Email body."
                            ),
                        },
                    },
                    "required": [
                        "to",
                        "subject",
                        "body",
                    ],
                },
            },
        }

    def _get_credentials(self):

        creds = None

        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file(
                "token.json",
                SCOPES,
            )

        if not creds or not creds.valid:

            if (
                creds
                and creds.expired
                and creds.refresh_token
            ):
                creds.refresh(Request())

            else:
                flow = (
                    InstalledAppFlow
                    .from_client_secrets_file(
                        "credentials.json",
                        SCOPES,
                    )
                )

                creds = flow.run_local_server(
                    port=0
                )

            with open("token.json", "w") as token:
                token.write(creds.to_json())

        return creds

    def run(
        self,
        to: str,
        subject: str,
        body: str,
    ):

        credentials = self._get_credentials()

        service = build(
            "gmail",
            "v1",
            credentials=credentials,
        )

        message = EmailMessage()

        message.set_content(body)

        message["To"] = to
        message["Subject"] = subject

        encoded_message = (
            base64.urlsafe_b64encode(
                message.as_bytes()
            )
            .decode()
        )

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

        return (
            f"Email sent successfully. "
            f"Message ID: {result['id']}"
        )