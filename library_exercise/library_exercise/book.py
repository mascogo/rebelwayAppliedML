import json
from library_exercise.user import User
from dataclasses import dataclass, field

# I'm not use 
class Book:
    def __init__(self, title, author, isbn, checked_out=False, checked_out_by=None):
        self.title = title
        self.author = author
        self.isbn = isbn
        self.checked_out = checked_out
        self.checked_out_by = checked_out_by  # user_id or None

    def __repr__(self):
        return f"<Book {self.title} by {self.author} (ISBN: {self.isbn})>"

    def to_dict(self):
        return {
            self.isbn: {
                "title": self.title,
                "author": self.author,
                "isbn": self.isbn,
                "checked_out": self.checked_out,
                "checked_out_by": self.checked_out_by.user_id if self.checked_out_by else None
            }
        }

    @staticmethod
    def from_dict(data, users):
        checked_out_by =  next((u_id for u_id in users if u_id == data["checked_out_by"]), None)
        return Book(
            data["title"],
            data["author"],
            data["isbn"],
            data["checked_out"],
            checked_out_by
        )