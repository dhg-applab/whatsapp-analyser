from enum import Enum


class MessageType(Enum):
    TEXT = 1
    STICKER = 2
    VOICE_MESSAGE = 3
    PHOTO = 4
    VIEW_ONCE_PHOTO = 5
    VIDEO = 6
    VIEW_ONCE_VIDEO = 7
    FILE = 8
    LOCATION = 9
    CONTACT = 10
    POLL = 11
