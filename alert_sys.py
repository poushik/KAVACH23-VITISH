from twilio.rest import Client

import keys
client = Client(keys.account_sid,keys.auth_token)


def send_message():
    client.messages.create(
         body="High alert",
         from_=keys.twilio_number,
         to=keys.target_no)

def send_call():
    client.calls.create(
        twiml='<Response><Say><high alert></Say></Response>',
        from_=keys.twilio_number,
        to=keys.target_no

    )
