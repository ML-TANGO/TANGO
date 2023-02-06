import json

import sqlalchemy
from sqlalchemy.types import TypeDecorator

SIZE = 4096


class TextPickleType(TypeDecorator):

    impl = sqlalchemy.Text(SIZE)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            value = json.dumps(value)

        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            value = json.loads(value)
        return value
