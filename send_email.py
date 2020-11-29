from __future__ import print_function
import httplib2
import os
import base64
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
import mimetypes
from googleapiclient import discovery, errors
import oauth2client
from oauth2client import client, tools, file
import json

try:
    import argparse
    flags = argparse.ArgumentParser(parents=[tools.argparser]).parse_args()
except ImportError:
    flags = None

SCOPES = "https://mail.google.com/"
CLIENT_SECRET_FILE = 'credentials.json'
APPLICATION_NAME = 'Security Camera'


def get_credentials():
    """Gets valid user credentials from storage.

    If nothing has been stored, or if the stored credentials are invalid,
    the OAuth2 flow is completed to obtain the new credentials.

    Returns:
      Credentials, the obtained credential.
    """
    home_dir = os.path.expanduser('~')
    credential_dir = os.path.join(home_dir, '.credentials')
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)
    credential_path = os.path.join(credential_dir,
                                             'gmail-python-quickstart.json')

    store = oauth2client.file.Storage(credential_path)
    credentials = store.get()
    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
        flow.user_agent = APPLICATION_NAME
        if flags:
             credentials = tools.run_flow(flow, store, flags)
        else: # Needed only for compatibility with Python 2.6
             credentials = tools.run(flow, store)
        print('Storing credentials to ' + credential_path)
    return credentials

# create a message
def CreateMessage(sender, to, subject, message_text):
    """Create a message for an email.
      Args:
      sender: Email address of the sender.
      to: Email address of the receiver.
      subject: The subject of the email message.
      message_text: The text of the email message.

      Returns:
      An object containing a base64 encoded email object.
    """
    message = MIMEText(message_text)
    #message = message_text
    message['to'] = to
    message['from'] = sender
    message['subject'] = subject
    return {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}


def create_message_with_attachment(
    sender, to, subject, message_text, file):
  """Create a message for an email.

  Args:
    sender: Email address of the sender.
    to: Email address of the receiver.
    subject: The subject of the email message.
    message_text: The text of the email message.
    file: The path to the file to be attached.

  Returns:
    An object containing a base64url encoded email object.
  """
  message = MIMEMultipart()
  message['to'] = to
  message['from'] = sender
  message['subject'] = subject

  msg = MIMEText(message_text)
  message.attach(msg)

  content_type, encoding = mimetypes.guess_type(file)

  if content_type is None or encoding is not None:
    content_type = 'application/octet-stream'
  main_type, sub_type = content_type.split('/', 1)
  if main_type == 'text':
    fp = open(file, 'rb')
    msg = MIMEText(fp.read(), _subtype=sub_type)
    fp.close()
  elif main_type == 'image':
    fp = open(file, 'rb')
    msg = MIMEImage(fp.read(), _subtype=sub_type)
    fp.close()
  elif main_type == 'audio':
    fp = open(file, 'rb')
    msg = MIMEAudio(fp.read(), _subtype=sub_type)
    fp.close()
  else:
    fp = open(file, 'rb')
    msg = MIMEBase(main_type, sub_type)
    msg.set_payload(fp.read())
    fp.close()
  filename = os.path.basename(file)
  msg.add_header('Content-Disposition', 'attachment', filename=filename)
  message.attach(msg)

  return {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}


#send message
def SendMessage(service, user_id, message):
    """Send an email message.
    Args:
    service: Authorized Gmail API service instance.
    user_id: User's email address. The special value "me"
    can be used to indicate the authenticated user.
    message: Message to be sent.

    Returns:
    Sent Message.
    """
    try:
        print("I made it here")
        message = (service.users().messages().send(userId=user_id, body=message).execute())
        print ('Message Id: %s' % message['id'])
        return message
    except errors.HttpError as error:
        print ('An error occurred: %s' % error)


def SendEmail():
    """
    Shows basic usage of the Gmail API.
    Send a mail using gmail API
    """

    print("Sending text message")
    with open("C:\\Users\\pheig\\Documents\\GitHub\\security\\config_test.json") as json_data_file:
        data = json.load(json_data_file)
        email = data['email']
        sender = email['sender']
        to = email['receiver']
        subject = email['subject']

        credentials = get_credentials()
        http = credentials.authorize(httplib2.Http())
        service = discovery.build('gmail', 'v1', http=http)
        file = 'C:\\Users\\pheig\\Documents\\GitHub\\video\\28.11.2020.14.47.01\\frame10.jpg'
        msg_body = "test message"

        #message = CreateMessage("pheight2049@gmail.com", "EmperorShane@gmail.com", "Subject of email", msg_body)

        message = create_message_with_attachment(sender, to, subject, msg_body, file)

        SendMessage(service, "me", message)

def main():
    SendEmail()

if __name__ == '__main__':
    main()