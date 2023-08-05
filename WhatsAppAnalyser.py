import emoji
import langid
import nltk
import nltk.langnames as lgn
import pandas as pd
from datetime import datetime, timedelta
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pathlib import Path
from torchvision.io import read_image
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from transformers import pipeline
from typing import List, Tuple
from MessageType import MessageType
from WhatsAppExtractor import WhatsAppExtractor

nltk.download('bcp47')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


class WhatsAppAnalyser:
    """A class to analyse extracted WhatsApp chat data."""

    def __init__(self, chat_data: pd.DataFrame = None, chat_data_path: str = None, language: str = None):
        """Initialise WhatsAppAnalyser instance."""
        if chat_data and chat_data_path:
            raise ValueError('chat_data and chat_data_path cannot be set at the same time')
        if chat_data:
            self.chat_data_path = None
            self.chat_data = chat_data
        elif chat_data_path:
            self.chat_data_path = chat_data_path
            extractor = WhatsAppExtractor(include_emojis=True)
            self.chat_data = extractor.extract_chat_data(chat_data_path)
        else:
            raise ValueError('chat_data or chat_data_path must be set')
        if language:
            self.language = language
        else:
            self.language = self.detect_language(self.chat_data)

    @staticmethod
    def detect_language(chat_data, sample_ratio=0.1, min_sample_size=100) -> str:
        """Detect the language of the chat."""
        num_messages = len(chat_data)
        sample_size = min(num_messages, max(round(num_messages * sample_ratio), min_sample_size))
        sample = chat_data.sample(sample_size)
        language = langid.classify(sample['message'].str.cat(sep=' '))[0]
        return lgn.langname(language)

    def set_language(self, language: str):
        """Set the language of the chat."""
        self.language = language

    def _filter_messages(self,
                         user: str = None,
                         message_type: MessageType = None,
                         start_time: datetime = None,
                         end_time: datetime = None) -> pd.DataFrame:
        """Filter messages based on user, message type, start time, and end time."""
        filtered_data = self.chat_data.copy()

        if user:
            if not self._is_user_in_chat(user):
                raise ValueError(f'User {user} does not exist in the chat')
            filtered_data = filtered_data[filtered_data['user'] == user]
        if message_type:
            filtered_data = filtered_data[filtered_data['type'] == message_type]
        if start_time:
            filtered_data = filtered_data[filtered_data['datetime'] >= start_time]
        if end_time:
            if start_time and end_time < start_time:
                raise ValueError('end_time should be equal to or greater than start_time')
            filtered_data = filtered_data[filtered_data['datetime'] <= end_time]

        return filtered_data

    def _is_user_in_chat(self, user: str) -> bool:
        """Check if a user is in the chat."""
        return user in self.chat_data['user'].unique()

    def day_count(self,
                  user: str = None,
                  message_type: MessageType = None,
                  start_time: datetime = None,
                  end_time: datetime = None) -> int:
        """Count the number of days."""
        filtered_data = self._filter_messages(user, message_type, start_time, end_time)
        day_count = filtered_data['datetime'].dt.date.nunique()
        return day_count

    def day_count_by_user(self,
                          message_type: MessageType = None,
                          start_time: datetime = None,
                          end_time: datetime = None) -> pd.DataFrame:
        """Count the number of days by user."""
        filtered_data = self._filter_messages(message_type=message_type, start_time=start_time, end_time=end_time)
        filtered_data['date'] = filtered_data['datetime'].dt.date
        user_day_count = filtered_data.groupby('user')['date'].nunique().reset_index(name='day_count')
        return user_day_count

    def message_count(self,
                      frequency: str = None,
                      user: str = None,
                      message_type: MessageType = None,
                      start_time: datetime = None,
                      end_time: datetime = None) -> int | pd.DataFrame:
        """
        Count the number of messages.
        Frequency:
            Y: Year
            M: Month
            W: Week
            D: Day
            DOW: Day of week
            H: Hour
        """
        filtered_data = self._filter_messages(user, message_type, start_time, end_time)
        if frequency:
            match frequency:
                case 'Y':
                    return WhatsAppAnalyser.calculate_message_count_by_year(filtered_data)
                case 'M':
                    return WhatsAppAnalyser.calculate_message_count_by_month(filtered_data)
                case 'W':
                    return WhatsAppAnalyser.calculate_message_count_by_week(filtered_data)
                case 'D':
                    return WhatsAppAnalyser.calculate_message_count_by_day(filtered_data)
                case 'DOW':
                    return WhatsAppAnalyser.calculate_message_count_by_day_of_week(filtered_data)
                case 'H':
                    return WhatsAppAnalyser.calculate_message_count_by_hour(filtered_data)
                case _:
                    raise ValueError(f'Invalid frequency: {frequency}')
        else:
            return len(filtered_data)

    def message_count_by_user(self,
                              frequency: str = None,
                              message_type: MessageType = None,
                              start_time: datetime = None,
                              end_time: datetime = None) -> pd.DataFrame:
        """
        Count the number of messages by user.
        Frequency:
            Y: Year
            M: Month
            W: Week
            D: Day
            DOW: Day of week
            H: Hour
        """
        filtered_data = self._filter_messages(message_type=message_type, start_time=start_time, end_time=end_time)
        if frequency:
            match frequency:
                case 'Y':
                    func = WhatsAppAnalyser.calculate_message_count_by_year
                case 'M':
                    func = WhatsAppAnalyser.calculate_message_count_by_month
                case 'W':
                    func = WhatsAppAnalyser.calculate_message_count_by_week
                case 'D':
                    func = WhatsAppAnalyser.calculate_message_count_by_day
                case 'DOW':
                    func = WhatsAppAnalyser.calculate_message_count_by_day_of_week
                case 'H':
                    func = WhatsAppAnalyser.calculate_message_count_by_hour
                case _:
                    raise ValueError(f'Invalid frequency: {frequency}')
            return filtered_data.groupby('user').apply(func)
        else:
            return filtered_data['user'].value_counts().reset_index(name='message_count')

    @staticmethod
    def calculate_message_count_by_year(chat_data: pd.DataFrame) -> pd.DataFrame:
        """Count the number of messages by year."""
        message_count_by_year = chat_data.groupby(
            pd.Grouper(key='datetime', freq='Y')
        ).size().reset_index(name='message_count')
        message_count_by_year['year'] = message_count_by_year['datetime'].dt.year
        message_count_by_year = message_count_by_year.drop(columns='datetime')
        message_count_by_year['message_count'] = message_count_by_year.pop('message_count')
        return message_count_by_year

    @staticmethod
    def calculate_message_count_by_month(chat_data: pd.DataFrame) -> pd.DataFrame:
        """Count the number of messages by month."""
        message_count_by_month = chat_data.groupby(
            pd.Grouper(key='datetime', freq='M')
        ).size().reset_index(name='message_count')
        message_count_by_month['year_month'] = message_count_by_month['datetime'].dt.to_period('M')
        message_count_by_month['month'] = message_count_by_month['datetime'].dt.month_name()
        message_count_by_month = message_count_by_month.drop(columns='datetime')
        message_count_by_month['message_count'] = message_count_by_month.pop('message_count')
        return message_count_by_month

    @staticmethod
    def calculate_message_count_by_week(chat_data: pd.DataFrame) -> pd.DataFrame:
        """Count the number of messages by week."""
        message_count_by_week = chat_data.groupby(
            pd.Grouper(key='datetime', freq='W')
        ).size().reset_index(name='message_count')
        message_count_by_week['year'] = message_count_by_week['datetime'].dt.year
        message_count_by_week['week'] = message_count_by_week['datetime'].dt.isocalendar().week
        message_count_by_week = message_count_by_week.drop(columns='datetime')
        message_count_by_week['message_count'] = message_count_by_week.pop('message_count')
        return message_count_by_week

    @staticmethod
    def calculate_message_count_by_day(chat_data: pd.DataFrame) -> pd.DataFrame:
        """Count the number of messages by day."""
        message_count_by_day = chat_data.groupby(
            pd.Grouper(key='datetime', freq='D')
        ).size().reset_index(name='message_count')
        message_count_by_day['year_month_day'] = message_count_by_day['datetime'].dt.to_period('D')
        message_count_by_day['year'] = message_count_by_day['datetime'].dt.year
        message_count_by_day['month'] = message_count_by_day['datetime'].dt.month
        message_count_by_day['day'] = message_count_by_day['datetime'].dt.day
        message_count_by_day = message_count_by_day.drop(columns='datetime')
        message_count_by_day['message_count'] = message_count_by_day.pop('message_count')
        return message_count_by_day

    @staticmethod
    def calculate_message_count_by_day_of_week(chat_data: pd.DataFrame) -> pd.DataFrame:
        """Count the number of messages by day of week."""
        chat_data['day_of_week'] = chat_data['datetime'].dt.day_of_week
        chat_data['day_name'] = chat_data['datetime'].dt.day_name()
        message_count_by_day_of_week = chat_data.groupby(['day_of_week', 'day_name']).size() \
            .reset_index(name='message_count')
        return message_count_by_day_of_week

    @staticmethod
    def calculate_message_count_by_hour(chat_data: pd.DataFrame) -> pd.DataFrame:
        """Count the number of messages by hour."""
        message_count_by_hour = chat_data.groupby(
            pd.Grouper(key='datetime', freq='H')
        ).size().reset_index(name='message_count')
        message_count_by_hour['year'] = message_count_by_hour['datetime'].dt.year
        message_count_by_hour['month'] = message_count_by_hour['datetime'].dt.month
        message_count_by_hour['day'] = message_count_by_hour['datetime'].dt.day
        message_count_by_hour['hour'] = message_count_by_hour['datetime'].dt.hour
        message_count_by_hour['hour_period'] = message_count_by_hour['datetime'].dt.to_period('H')
        message_count_by_hour = message_count_by_hour.drop(columns='datetime')
        message_count_by_hour['message_count'] = message_count_by_hour.pop('message_count')
        return message_count_by_hour

    def user_count(self) -> int:
        """Count the number of users."""
        return self.chat_data['user'].nunique()

    @staticmethod
    def tokenize(messages: pd.Series,
                 language: str = 'English',
                 remove_stop_words: bool = False,
                 stemming: bool = False,
                 lemmatization: bool = False) -> List[str]:
        """Tokenize the messages into words."""
        if stemming and lemmatization:
            raise ValueError('stemming and lemmatization cannot be set at the same time')

        # Remove non-alphanumeric characters and convert to lowercase
        messages = messages.astype(str).replace('[^a-zA-Z0-9 \']', ' ', regex=True).str.lower()

        # Concatenate all messages
        messages = messages.str.cat()

        # Tokenize the messages into words
        words = word_tokenize(messages)

        # Remove stop words
        if remove_stop_words:
            stop_words = set(stopwords.words(language))
            words = [word for word in words if word not in stop_words]

        # Stemming
        if stemming:
            ps = PorterStemmer()
            words = [ps.stem(word) for word in words]

        # Lemmatization
        if lemmatization:
            wnl = WordNetLemmatizer()
            words = [wnl.lemmatize(word, pos='v') for word in words]

        return words

    def word_count(self,
                   remove_stop_words: bool = False,
                   user: str = None,
                   start_time: datetime = None,
                   end_time: datetime = None) -> int:
        """Count the number of words in the text messages."""
        filtered_data = self._filter_messages(user, MessageType.TEXT, start_time, end_time)
        words = pd.Series(WhatsAppAnalyser.tokenize(
            filtered_data['message'], self.language, remove_stop_words)
        )
        return len(words)

    def word_count_by_user(self,
                           remove_stop_words: bool = False,
                           start_time: datetime = None,
                           end_time: datetime = None) -> pd.DataFrame:
        """Count the number of words in the text messages by user."""
        filtered_data = self._filter_messages(message_type=MessageType.TEXT, start_time=start_time, end_time=end_time)
        user_word_count = filtered_data.groupby('user').apply(
            lambda x: len(pd.Series(WhatsAppAnalyser.tokenize(
                filtered_data['message'], self.language, remove_stop_words)
            ))
        ).reset_index(name='word_count')
        return user_word_count

    def unique_word_count(self,
                          remove_stop_words: bool = False,
                          stemming: bool = False,
                          lemmatization: bool = False,
                          user: str = None,
                          start_time: datetime = None,
                          end_time: datetime = None) -> int:
        """Count the number of unique words in the text messages."""
        filtered_data = self._filter_messages(user, MessageType.TEXT, start_time, end_time)
        words = pd.Series(WhatsAppAnalyser.tokenize(
            filtered_data['message'], self.language, remove_stop_words, stemming, lemmatization)
        )
        return words.nunique()

    def unique_word_count_by_user(self,
                                  remove_stop_words: bool = False,
                                  stemming: bool = False,
                                  lemmatization: bool = False,
                                  start_time: datetime = None,
                                  end_time: datetime = None) -> pd.DataFrame:
        """Count the number of unique words in the text messages by user."""
        filtered_data = self._filter_messages(message_type=MessageType.TEXT, start_time=start_time, end_time=end_time)
        user_unique_word_count = filtered_data.groupby('user').apply(
            lambda x: pd.Series(WhatsAppAnalyser.tokenize(
                filtered_data['message'], self.language, remove_stop_words, stemming, lemmatization)
            ).nunique()
        ).reset_index(name='unique_word_count')
        return user_unique_word_count

    @staticmethod
    def calculate_most_common_words(chat_data: pd.DataFrame,
                                    n: int = 10,
                                    language: str = 'English',
                                    remove_stop_words: bool = False,
                                    stemming: bool = False,
                                    lemmatization: bool = False) -> pd.DataFrame:
        """Calculate the most common n words in the text messages."""
        # Tokenize the messages into words
        words = WhatsAppAnalyser.tokenize(chat_data['message'], language, remove_stop_words, stemming, lemmatization)

        # Calculate the frequency distribution of words
        word_freq = FreqDist(words)

        # Get the most common n words
        most_common_words = word_freq.most_common(n)

        return pd.DataFrame(most_common_words, columns=['word', 'count'])

    def most_common_words(self,
                          n: int = 10,
                          remove_stop_words: bool = False,
                          stemming: bool = False,
                          lemmatization: bool = False,
                          user: str = None,
                          start_time: datetime = None,
                          end_time: datetime = None) -> pd.DataFrame:
        """Calculate the most common n words in the text messages."""
        filtered_data = self._filter_messages(user, MessageType.TEXT, start_time, end_time)
        most_common_words = WhatsAppAnalyser.calculate_most_common_words(
            filtered_data, n, self.language, remove_stop_words, stemming, lemmatization
        )
        return most_common_words

    def most_common_words_by_user(self,
                                  n: int = 10,
                                  remove_stop_words: bool = False,
                                  stemming: bool = False,
                                  lemmatization: bool = False,
                                  start_time: datetime = None,
                                  end_time: datetime = None) -> pd.DataFrame:
        """Calculate the most common n words in the text messages by user."""
        filtered_data = self._filter_messages(message_type=MessageType.TEXT, start_time=start_time, end_time=end_time)
        user_most_common_words = filtered_data.groupby('user').apply(
            lambda x: WhatsAppAnalyser.calculate_most_common_words(x, n, self.language, remove_stop_words, stemming, lemmatization)
        )
        return user_most_common_words

    @staticmethod
    def filter_emojis(message: str) -> str:
        """Filter emojis from message."""
        emojis = set(emoji.EMOJI_DATA.keys())
        filtered_message = ''.join([c for c in message if c in emojis])
        return filtered_message if filtered_message else None

    @staticmethod
    def calculate_most_common_emojis(chat_data: pd.DataFrame,
                                     n: int = 10) -> pd.DataFrame:
        """Calculate the most common n emojis in the text messages."""
        emojis = chat_data['message'].apply(WhatsAppAnalyser.filter_emojis).dropna()
        most_common_emojis = emojis.value_counts().head(n).reset_index(name='count').rename(
            columns={'message': 'emoji'}
        )
        return most_common_emojis

    def most_common_emojis(self,
                           n: int = 10,
                           user: str = None,
                           start_time: datetime = None,
                           end_time: datetime = None) -> pd.DataFrame:
        """Get the most common n emojis in the text messages."""
        filtered_data = self._filter_messages(user, MessageType.TEXT, start_time, end_time)
        most_common_emojis = WhatsAppAnalyser.calculate_most_common_emojis(filtered_data, n)
        return most_common_emojis

    def most_common_emojis_by_user(self,
                                   n: int = 10,
                                   start_time: datetime = None,
                                   end_time: datetime = None) -> pd.DataFrame:
        """Get the most common n emojis in the text messages by user."""
        filtered_data = self._filter_messages(message_type=MessageType.TEXT, start_time=start_time, end_time=end_time)
        user_most_common_emojis = filtered_data.groupby('user').apply(
            lambda x: WhatsAppAnalyser.calculate_most_common_emojis(x, n)
        )
        return user_most_common_emojis

    @staticmethod
    def calculate_longest_message(chat_data: pd.DataFrame) -> Tuple[str, int]:
        """Calculate the longest message and its length."""
        messages = chat_data['message'].reset_index()
        longest_message_idx = messages['message'].map(len).idxmax()
        longest_message = messages.iloc[longest_message_idx, messages.columns.get_loc('message')]
        return longest_message, len(longest_message)

    def longest_message(self,
                        user: str = None,
                        start_time: datetime = None,
                        end_time: datetime = None) -> Tuple[str, int]:
        """Calculate the longest message and its length."""
        filtered_data = self._filter_messages(user, MessageType.TEXT, start_time, end_time)
        longest_message, length = WhatsAppAnalyser.calculate_longest_message(filtered_data)
        return longest_message, length

    def longest_message_by_user(self,
                                start_time: datetime = None,
                                end_time: datetime = None) -> pd.DataFrame:
        """Calculate the longest message and its length by user."""
        filtered_data = self._filter_messages(message_type=MessageType.TEXT, start_time=start_time, end_time=end_time)
        users = self.chat_data['user'].unique()
        user_longest_message = {}
        for user in users:
            user_filtered_data = filtered_data[filtered_data['user'] == user]
            longest_message, length = WhatsAppAnalyser.calculate_longest_message(user_filtered_data)
            user_longest_message[user] = {'longest_message': longest_message, 'length': length}
        return pd.DataFrame.from_dict(user_longest_message, orient='index')

    @staticmethod
    def calculate_chat_duration(chat_data: pd.DataFrame) -> Tuple[datetime, datetime, timedelta]:
        """Calculate the duration of the chat."""
        start_datetime = chat_data['datetime'].min()
        end_datetime = chat_data['datetime'].max()
        duration = end_datetime - start_datetime
        return start_datetime, end_datetime, duration

    def chat_duration(self,
                      user: str = None,
                      message_type: MessageType = None,
                      start_time: datetime = None,
                      end_time: datetime = None) -> Tuple[datetime, datetime, timedelta]:
        """Calculate the duration of the chat."""
        filtered_data = self._filter_messages(user, message_type, start_time, end_time)
        start_datetime, end_datetime, duration = WhatsAppAnalyser.calculate_chat_duration(filtered_data)
        return start_datetime, end_datetime, duration

    def chat_duration_by_user(self,
                              message_type: MessageType = None,
                              start_time: datetime = None,
                              end_time: datetime = None) -> pd.DataFrame:
        """Calculate the duration of the chat by user."""
        filtered_data = self._filter_messages(message_type=message_type, start_time=start_time, end_time=end_time)
        users = self.chat_data['user'].unique()
        user_chat_duration = {}
        for user in users:
            user_filtered_data = filtered_data[filtered_data['user'] == user]
            start_datetime, end_datetime, duration = WhatsAppAnalyser.calculate_chat_duration(user_filtered_data)
            user_chat_duration[user] = {
                'start': start_datetime,
                'end': end_datetime,
                'duration': duration
            }
        return pd.DataFrame.from_dict(user_chat_duration, orient='index')

    def average_message_length(self,
                               user: str = None,
                               start_time: datetime = None,
                               end_time: datetime = None) -> float:
        """Calculate the average length of messages."""
        filtered_data = self._filter_messages(user, MessageType.TEXT, start_time, end_time)
        avg_length = filtered_data['message'].str.len().mean()
        return avg_length

    def average_message_length_by_user(self,
                                       start_time: datetime = None,
                                       end_time: datetime = None) -> pd.DataFrame:
        """Calculate the average length of messages by user."""
        filtered_data = self._filter_messages(message_type=MessageType.TEXT, start_time=start_time, end_time=end_time)
        user_avg_length = filtered_data.groupby('user')['message'].apply(
            lambda x: x.str.len().mean()
        ).reset_index(name='average_length')
        return user_avg_length

    def average_time_between_messages(self,
                                      user: str = None,
                                      start_time: datetime = None,
                                      end_time: datetime = None) -> timedelta:
        """Calculate the average time between two messages."""
        filtered_data = self._filter_messages(user, MessageType.TEXT, start_time, end_time)
        sorted_data = filtered_data.sort_values('datetime')
        avg_time_diff = sorted_data['datetime'].diff().mean()
        return avg_time_diff

    def average_time_between_messages_by_user(self,
                                              start_time: datetime = None,
                                              end_time: datetime = None) -> pd.DataFrame:
        """Calculate the average time between two messages by user."""
        filtered_data = self._filter_messages(message_type=MessageType.TEXT, start_time=start_time, end_time=end_time)
        sorted_data = filtered_data.sort_values(['user', 'datetime'])
        time_diff = sorted_data.groupby('user')['datetime'].diff()
        avg_time_diff = time_diff.groupby(sorted_data['user']).mean().reset_index(name='average_time_diff')
        return avg_time_diff

    @staticmethod
    def calculate_average_reply_time(chat_data: pd.DataFrame, user: str) -> timedelta:
        """Calculate the average time the user takes to reply to a message."""
        reply_time = timedelta()
        num_replies = 0
        last_message_time = None
        replied = False
        for _, row in chat_data.iterrows():
            if row['user'] == user:
                if not replied and last_message_time:
                    reply_time += row['datetime'] - last_message_time
                    num_replies += 1
                    replied = True
            else:
                last_message_time = row['datetime']
                replied = False
        if num_replies == 0:
            raise ValueError(f'No reply found for the user {user}')
        return reply_time / num_replies

    def average_reply_time(self,
                           user: str,
                           start_time: datetime = None,
                           end_time: datetime = None) -> timedelta:
        """Calculate the average time the user takes to reply to a message."""
        if not self._is_user_in_chat(user):
            raise ValueError(f'User {user} does not exist in the chat')
        filtered_data = self._filter_messages(start_time=start_time, end_time=end_time)
        average_reply_time = WhatsAppAnalyser.calculate_average_reply_time(filtered_data, user)
        return average_reply_time

    def average_reply_time_by_user(self,
                                   start_time: datetime = None,
                                   end_time: datetime = None) -> pd.DataFrame:
        """Calculate the average time the user takes to reply to a message by user."""
        filtered_data = self._filter_messages(start_time=start_time, end_time=end_time)
        users = self.chat_data['user'].unique()
        user_average_reply_time = {}
        for user in users:
            try:
                user_average_reply_time[user] = WhatsAppAnalyser.calculate_average_reply_time(filtered_data, user)
            except ValueError:
                continue
        return pd.DataFrame.from_dict(user_average_reply_time, orient='index',
                                      columns=['average_reply_time']).reset_index().rename(columns={'index': 'user'})

    @staticmethod
    def image_classification(image_path: str) -> str:
        """Classify the image."""
        # Read image file
        img = read_image(image_path)

        # Initialise the model
        weights = EfficientNet_V2_S_Weights.DEFAULT
        model = efficientnet_v2_s(weights=weights)
        model.eval()

        # Initialise the inference transforms
        preprocess = weights.transforms()

        # Apply inference preprocessing transforms
        batch = preprocess(img).unsqueeze(0)

        # Perform prediction
        prediction = model(batch).squeeze(0).softmax(0)
        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
        category_name = weights.meta["categories"][class_id]

        return category_name

    def classify_photos(self,
                        user: str = None,
                        start_time: datetime = None,
                        end_time: datetime = None) -> pd.DataFrame:
        """Classify photos in the chat."""
        if self.chat_data_path is None:
            raise ValueError('To classify photos, the class must be initialised with chat_data_path')
        filtered_data = self._filter_messages(user, MessageType.PHOTO, start_time, end_time).reset_index(drop=True)
        filtered_data['photo_category'] = filtered_data['message'].apply(
            lambda filename: WhatsAppAnalyser.image_classification(str(Path(self.chat_data_path, filename)))
        )
        return filtered_data

    @staticmethod
    def image_captioning(image_paths: List[str], max_tokens: int) -> List[str]:
        """Caption the images."""
        captioner = pipeline(model='Salesforce/blip-image-captioning-base')
        captions = captioner(image_paths, max_new_tokens=max_tokens)
        captions = [caption[0]['generated_text'] for caption in captions]
        return captions

    def caption_photos(self,
                       max_tokens: int = 20,
                       user: str = None,
                       start_time: datetime = None,
                       end_time: datetime = None) -> pd.DataFrame:
        """Caption photos in the chat."""
        if self.chat_data_path is None:
            raise ValueError('To caption photos, the class must be initialised with chat_data_path')
        filtered_data = self._filter_messages(user, MessageType.PHOTO, start_time, end_time).reset_index(drop=True)
        image_paths = filtered_data['message'].apply(
            lambda filename: str(Path(self.chat_data_path, filename))
        ).tolist()
        captions = WhatsAppAnalyser.image_captioning(image_paths, max_tokens)
        filtered_data['photo_caption'] = pd.Series(captions)
        return filtered_data

    @staticmethod
    def offensive_language_identification(messages: List[str]) -> Tuple[List[str], List[float]]:
        """Identify offensive language in the messages."""
        classifier = pipeline(model='cardiffnlp/twitter-roberta-base-dec2021-offensive')
        predictions = classifier(messages)
        labels, scores = zip(*map(lambda x: x.values(), predictions))
        return labels, scores

    def identify_offensive_language(self,
                                    user: str = None,
                                    start_time: datetime = None,
                                    end_time: datetime = None) -> pd.DataFrame:
        """Identify offensive language in the messages."""
        if self.language != 'English':
            raise ValueError('Offensive language identification is only supported for English chats.')
        filtered_data = self._filter_messages(user, MessageType.TEXT, start_time, end_time).reset_index(drop=True)
        labels, scores = WhatsAppAnalyser.offensive_language_identification(filtered_data['message'].tolist())
        filtered_data['offensive_language_label'] = pd.Series(labels)
        filtered_data['offensive_language_score'] = pd.Series(scores)
        return filtered_data
