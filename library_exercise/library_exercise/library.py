import os
import json
from library_exercise.book import Book
from library_exercise.user import User

class Library:
    def __init__(self, database_file):
        self.books = {}
        self.users = {}
        self.db_file = database_file
        if not os.path.isfile(database_file):
            self.save_to_json()

    def add_book(self, book):
        if book.isbn in self.books:
            print("Warning: Book already registered with the same ISBN")
            return False
        self.books.update(book.to_dict())
        self.save_to_json()
        return True

    def add_user(self, user):
        if user.user_id in self.users:
            print("User_id '{}' is already registered to '{}'".format(user.user_id, self.users[user_id]["name"]))
            return False
        self.users.update(user.to_dict())
        self.save_to_json()
        return True

    def checkout_book(self, book, user):
        print("will checkout book: '{}' by user: '{}'".format(book, user))
        if book.checked_out:
            print("WARNING: Book '{}' is already checked out by user '{}'".format(book, book.checked_out_by))
            return False

        if book and user:
            book.checked_out = True
            book.checked_out_by = user
            user.borrowed_books.append(book.isbn)
            self.books.update(book.to_dict())
            self.users.update(user.to_dict())
            self.save_to_json()
            return True

        return False

    def return_book(self, book, user):
        print("User '{}' is returning book '{}'".format(user, book))
        if user and book:
            book.checked_out = False
            book.checked_out_by = None
            user.borrowed_books.remove(book.isbn)
            self.books.update(book.to_dict())
            self.users.update(user.to_dict())
            self.save_to_json()
            return True
        return False

    def list_books(self):
        return self.books

    def list_users(self):
        return self.users

    def reset_database(self):
        self.books = {}
        self.users = {}
        self.save_to_json()

    def save_to_json(self):
        data = {
            "books": self.books,
            "users":  self.users
        }
        with open(self.db_file, "w") as f:
            json.dump(data, f, indent=2)

    def load_from_json(self):
        print("__load_from_json__")
        try:
            with open(self.db_file, "r") as f:
                data = json.load(f)
            self.books = dict([(b["isbn"], Book.from_dict(b, [])) for b in data["books"]])
            self.users = dict([(u["user_id"], User.from_dict(u, self.books)) for u in data["users"]])
            # Update checked_out_by references
            for isbn in self.books:
                if self.books[isbn].checked_out_by:
                    self.books[isbn].checked_out_by = next((u for u in self.users if u.user_id == self.books[isbn].checked_out_by.user_id), None)
        except FileNotFoundError:
            pass

