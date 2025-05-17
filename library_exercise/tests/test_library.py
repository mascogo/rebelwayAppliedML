import pytest
from library_exercise.library import Library
from library_exercise.book import Book
from library_exercise.user import User

@pytest.fixture
def library():
    database = "test_library_db.json"
    lib = Library(database)
    lib.reset_database()
    return lib

def test_add_user(library):
    user3 = User("Ally", 3)
    user4 = User("Bert", 4)
    library.add_user(user3)
    library.add_user(user4)
    assert len(library.list_users()) == 2

def test_add_book(library):
    book1 = Book("Star Wars", "Alan Dean Foster", "1111")
    book2 = Book("Star Wars", "George Lucas", "1138")
    library.add_book(book1)
    library.add_book(book2)
    assert len(library.list_books()) == 2

def test_reset_database(library):
    book= Book("Star Wars", "Alan Dean Foster", "unknown_isbn")
    library.add_book(book)
    library.reset_database()
    assert len(library.list_books()) == 0 and len(library.list_users()) == 0

def test_checkout_book(library):
    book1 = Book("Star Wars", "Alan Dean Foster", "unknown_isbn")
    book2 = Book("Star Wars", "George Lucas", "1138")
    library.add_book(book1)
    library.add_book(book2)

    user1 = User("Ally", 1)
    user2 = User("Bert", 2)
    library.add_user(user1)
    library.add_user(user2)

    assert library.checkout_book(book1, user1) == True
    assert library.checkout_book(book1, user2) == False