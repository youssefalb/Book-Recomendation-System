import torch
class BookDataset:
    def __init__(self, user_ids, isbns, ratings):
        self.user_ids = user_ids
        self.isbns = isbns
        self.ratings = ratings
        
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, item):
        try:
            user_id = self.user_ids[item]
            isbn = self.isbns[item]
            rating = self.ratings[item]
        except IndexError:
            print(f"IndexError: Invalid index '{item}'")
            return None
        except KeyError:
            print(f"KeyError: Invalid key '{item}'")
            return None
        return {
            "user_id": torch.tensor(user_id, dtype=torch.float),
            "isbn": torch.tensor(isbn, dtype=torch.float),
            "rating": torch.tensor(rating, dtype=torch.float),
        }