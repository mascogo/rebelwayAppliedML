import json 
from dataclasses import dataclass, field

class User:
    def __init__(self, name, user_id, borrowed_books=None):
        self.name = name
        self.user_id = user_id
        self.borrowed_books = borrowed_books if borrowed_books else []

    def __repr__(self):
        return f"<User {self.name} (ID: {self.user_id})>"

    def to_dict(self):
        return {
            self.user_id: {
                "name": self.name,
                "user_id": self.user_id,
                "borrowed_books": [book for book in self.borrowed_books]
            }
        }

    @staticmethod
    def from_dict(data, books):
        borrowed_books = [next((b_isbn for b_isbn in books if b_isbn == isbn), None) for isbn in data["borrowed_books"]]
        borrowed_books = [b for b in borrowed_books if b]
        return User(data["name"], data["user_id"], borrowed_books)