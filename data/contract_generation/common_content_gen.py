import random
import string

numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
street_types = ['Street', 'Road', 'Lane', 'Drive', 'Avenue', 'Boulevard', 'Court', 'Crescent', 'Highway', 'Terrace', 'Way', 'Place', 'Square', 'Alley', 'Circle']
company_types = ['Ltd', 'Inc', 'Corp', 'LLC', 'PLC', 'GmbH', 'AG', 'SA', 'SARL', 'SRL', 'BV', 'BVBA', 'NV', 'SE', 'SAS'] 


class Content_Gen:

    def __init__(self):
      self.address_num_len = 3
      self.address_st_len = 6
      self.given_name_len = 4
      self.surname_len = 4
      self.short_company_name = 6
      self.long_company_name = 40
      

    # Generate random address
    def address_gen(self):
      address = []    
      # Generate numbers part  
      for char in range(self.address_num_len):
        address.append(random.choice(numbers))
      address.append(' ')
      # Genereate letters part (street name)
      letters_part = ''.join(random.choices(string.ascii_lowercase, k=self.address_st_len))
      letters_part = letters_part.capitalize()  # Capitalize the first letter
      address.append(letters_part + ' ')
      # Genereate street type
      address.append(random.choice(street_types))      
      return ''.join(address)

    # Generate random name (given name + surname)
    def name_gen(self):
      name = []
      #Generate given name
      given_name_part = ''.join(random.choices(string.ascii_lowercase, k=self.given_name_len))
      given_name_part = given_name_part.capitalize()  # Capitalize the first letter
      name.append(given_name_part + ' ')

      #Generate surname
      surname_part = ''.join(random.choices(string.ascii_lowercase, k=self.surname_len))
      surname_part = surname_part.capitalize()  # Capitalize the first letter
      name.append(surname_part)
      return ''.join(name)

    # Generate random company name
    def company_name_gen(self, short_name=True):
      name = []
      if short_name:
        name_len = self.short_company_name
      else:
        name_len = self.long_company_name
      name_part = ''.join(random.choices(string.ascii_lowercase, k=name_len))
      name_part = name_part.capitalize()  # Capitalize the first letter
      name.append(name_part)
      name.append(' ')
      name.append(random.choice(company_types))
      return ''.join(name)

    # Generate random date in the format "dd-mm-yyyy"  
    def date_gen(self, start_year, end_year):
      year = random.randint(start_year, end_year)
      month = random.randint(1, 12)    
      # Generate random day based on the selected month
      if month in [1, 3, 5, 7, 8, 10, 12]:
        day = random.randint(1, 31)
      elif month in [4, 6, 9, 11]:
        day = random.randint(1, 30)
      else:  # February
        if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
            day = random.randint(1, 29)  # Leap year
        else:
            day = random.randint(1, 28)  # Non-leap year
      # Return the generated date as a string
      return f"{day:02d}-{month:02d}-{year}"

    # Generate random integer
    def int_gen(self, start, end, step):
      return random.randrange(start, end+1, step)
