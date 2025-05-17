from library_exercise.library import Library
from library_exercise.book import Book
from library_exercise.user import User

# Example usage:
if __name__ == "__main__":
    database_file = "library_db.json"
    library = Library(database_file)
    library.reset_database()
    library.load_from_json()
    print("List of books:\n{}".format(library.list_books()))
    book1 = Book("1984", "George Orwell", "1234567890")
    book2 = Book("Brave New World", "Aldous Huxley", "0987654321")
    user1 = User("Alice", 1)
    user2 = User("Bob", 2)

    library.add_book(book1)
    library.add_book(book2)
    library.add_user(user1)
    library.add_user(user2)

    print("List of books:\n{}".format(library.list_books()))
    print("user1.borrowed_books: {}".format(user1.borrowed_books))
    library.checkout_book(book1, user1)

    print("user1.borrowed_books: {}".format(user1.borrowed_books))
    library.checkout_book(book1, user2)
    library.return_book(book1, user1)
    print("user1.borrowed_books: {}".format(user1.borrowed_books))
    library.checkout_book(book1, user2)