import json
import random
import itertools

from common_content_gen import Content_Gen

content_gen = Content_Gen()

def convert_data(data):
    new_data = []
    # Iterate over each key in the dictionary, assuming each key represents an edge type
    for edge, items in data.items():
        for item in items:
            new_item = item.copy()  # Make a copy of each item
            new_item['edge'] = edge  # Add the 'edge' field with the current key as its value
            new_data.append(new_item)
    return new_data

# Load the input data
with open("data/contract_generation/data_input.json", "r") as json_file:
    data_input = json.load(json_file)

goods_items = data_input["goods_items"]
sale_parties = data_input["sale_parties"]
governing_law = data_input["governing_law"]

# Generate entity attributes
num_of_entity = 30
entity_attributes = {}
for entity in range(num_of_entity):
    entity_attributes[entity] = [content_gen.company_name_gen(short_name=True), content_gen.address_gen()]

# Sparse sub-graph: chain with 9 edges
entity_pairs_list_sparse = []
num_edge_sparse = 9
for i in range(num_edge_sparse):
     entity_pairs_list_sparse.append((i, i+1))

# Semi-dense sub-graph: 21 edges
entity_pairs_list_semi = []
for i in range(10, 19):
    entity_pairs_list_semi.append((i, i+1))

for i in range(4):
    entity_pairs_list_semi.append((10+i, 19-i))

for i in range(4):
    entity_pairs_list_semi.append((10+i, 18-i))

for i in range(4):
    entity_pairs_list_semi.append((14-i, 16+i))

# Dense: fully interconnected with 45 edges
entity_list = []
entity_pairs_list_dense = []
for i in range(20,30):
    entity_list.append(i)
combinations = list(itertools.combinations(entity_list, 2))
for i, combo in enumerate(combinations):
    entity_pairs_list_dense.append(combo)

entity_pairs_list = [entity_pairs_list_sparse, entity_pairs_list_semi, entity_pairs_list_dense]
entity_pairs = [tuple for sublist in entity_pairs_list for tuple in sublist]


class SalesQAGen():
    def __init__(self):
        self.start_year = 1980
        self.end_year = 2019
        self.min_invoice_after_delivery_days = 1
        self.max_invoice_after_delivery_days = 5
        self.ddl_start_days = 5
        self.ddl_end_days = 30
        self.ddl_step_days = 5
        self.min_cooling_off_days = 3
        self.max_cooling_off_days = 14
        self.min_warranty_years = 1
        self.max_warranty_years = 5
        self.min_default_interest_rate = 1
        self.max_default_interest_rate = 5
        self.entity_attributes = entity_attributes   

    # Generate sales contract attributes
    def sales_contract_QAgen(self, entity_pairs):
        sales_contract_attributes = []
        QA_sales_contract = {}
        for i in range(len(entity_pairs)):
            entity_pair = entity_pairs[i]
            seller_idx = random.choice(list(entity_pair))
            customer_idx = entity_pair[0] if entity_pair[1] == seller_idx else entity_pair[1]
            sales_contract_attributes.append(
                {
                    'contract_index': f"sales_contract_{i}",
                    'edge_note': str(entity_pair[0])+"_"+str(entity_pair[1]),
                    'effective_date': content_gen.date_gen(self.start_year, self.end_year),
                    'seller_name': entity_attributes[seller_idx][0],
                    'seller_address': entity_attributes[seller_idx][1],
                    'customer_name': entity_attributes[customer_idx][0],
                    'customer_address': entity_attributes[customer_idx][1],
                    'good': goods_items[i],
                    'quantity': random.randint(1, 100),
                    'price': random.randint(1, 100),
                    'invoice_provision_ddl_days': random.randint(self.min_invoice_after_delivery_days, self.max_invoice_after_delivery_days),
                    'invoice_ddl_days': random.randrange(self.ddl_start_days, self.ddl_end_days+1, self.ddl_step_days),
                    'payment_ddl_days': random.randrange(self.ddl_start_days, self.ddl_end_days+1, self.ddl_step_days),
                    'late_payment_interest_rate': random.randint(self.min_default_interest_rate, self.max_default_interest_rate),
                    'delivery_location': content_gen.address_gen(),
                    'shipping_method_decider': random.choice(sale_parties),
                    'payment_method_decider': random.choice(sale_parties),
                    'general_warranty_years': random.randint(self.min_warranty_years, self.max_warranty_years),
                    'notify_ddl_days': random.randrange(self.ddl_start_days, self.ddl_end_days+1, self.ddl_step_days),
                    'cooling_off_days': random.randint(self.min_cooling_off_days, self.max_cooling_off_days),
                    'governing_law': random.choice(governing_law),
                }
            )

            total_price = sales_contract_attributes[i]['quantity'] * sales_contract_attributes[i]['price']
            
            QA_sales_contract.update(
            {
            sales_contract_attributes[i]['edge_note']: 
            [
                {
                "question": f"What was the effective date of the contract between {sales_contract_attributes[i]['seller_name']} and {sales_contract_attributes[i]['customer_name']}?",
                "answer": f"{sales_contract_attributes[i]['effective_date']}."
                },
                {
                "question": f"What was the name of the seller in the contract with {sales_contract_attributes[i]['customer_name']} as of {sales_contract_attributes[i]['effective_date']}?",
                "answer": f"{sales_contract_attributes[i]['seller_name']}."
                },
                {
                "question": f"What was the address of {sales_contract_attributes[i]['seller_name']} in the contract with {sales_contract_attributes[i]['customer_name']}?",
                "answer": f"{sales_contract_attributes[i]['seller_address']}."
                },
                {
                "question": f"What was the name of the customer in the contract with {sales_contract_attributes[i]['seller_name']} as of {sales_contract_attributes[i]['effective_date']}?",
                "answer": f"{sales_contract_attributes[i]['customer_name']}."
                },
                {
                "question": f"What was the address of {sales_contract_attributes[i]['customer_name']} in the contract with {sales_contract_attributes[i]['seller_name']}?",
                "answer": f"{sales_contract_attributes[i]['customer_address']}."
                },
                {
                "question": f"What was the good that the seller was selling to the customer based on the contract between {sales_contract_attributes[i]['seller_name']} and {sales_contract_attributes[i]['customer_name']}?",
                "answer": f"{sales_contract_attributes[i]['good']}."
                },
                {
                "question": f"What was the quantity of the good being sold based on the contract between {sales_contract_attributes[i]['seller_name']} and {sales_contract_attributes[i]['customer_name']}?",
                "answer": f"{sales_contract_attributes[i]['quantity']}."
                },
                {
                "question": f"What was the unit price in dollars of the good being sold based on the contract between {sales_contract_attributes[i]['seller_name']} and {sales_contract_attributes[i]['customer_name']}?",
                "answer": f"{sales_contract_attributes[i]['price']}."
                },
                {
                "question": f"What was the total price in dollars of the good being sold based on the contract between {sales_contract_attributes[i]['seller_name']} and {sales_contract_attributes[i]['customer_name']}?",
                "answer": f"{total_price}."
                },
                {
                "question": f"By how many days after the delivery time must the seller provide the customer with an invoice based on the contract between {sales_contract_attributes[i]['seller_name']} and {sales_contract_attributes[i]['customer_name']}?",
                "answer": f"{sales_contract_attributes[i]['invoice_provision_ddl_days']}."
                },
                {
                "question": f"Within how many days must the invoice be paid in full based on the contract between {sales_contract_attributes[i]['seller_name']} and {sales_contract_attributes[i]['customer_name']}?",
                "answer": f"{sales_contract_attributes[i]['invoice_ddl_days']}."
                },
                {
                "question": f"After how many days would unpaid balances incur a late payment penalty based on the contract between {sales_contract_attributes[i]['seller_name']} and {sales_contract_attributes[i]['customer_name']}?",
                "answer": f"{sales_contract_attributes[i]['payment_ddl_days']}."
                },
                {
                "question": f"What was the late payment interest rate based on the contract between {sales_contract_attributes[i]['seller_name']} and {sales_contract_attributes[i]['customer_name']}?",
                "answer": f"{sales_contract_attributes[i]['late_payment_interest_rate']}%."
                },
                {
                "question": f"What was the address of delivery based on the contract between {sales_contract_attributes[i]['seller_name']} and {sales_contract_attributes[i]['customer_name']}?",
                "answer": f"{sales_contract_attributes[i]['delivery_location']}."
                },
                {
                "question": f"Who would decide the shipping method based on the contract between {sales_contract_attributes[i]['seller_name']} and {sales_contract_attributes[i]['customer_name']}?",
                "answer": f"{sales_contract_attributes[i]['shipping_method_decider']}."
                },
                {
                "question": f"Who would be responsible for the costs of the shipment based on the contract between {sales_contract_attributes[i]['seller_name']} and {sales_contract_attributes[i]['customer_name']}?",
                "answer": f"{sales_contract_attributes[i]['payment_method_decider']}."
                },
                {
                "question": f"What was the duration of the general warranty period in years based on the contract between {sales_contract_attributes[i]['seller_name']} and {sales_contract_attributes[i]['customer_name']}?",
                "answer": f"{sales_contract_attributes[i]['general_warranty_years']}."
                },
                {
                "question": f"Within how many days of discovering a defect must the customer notify the seller in writing in the event of a breach of warranty based on the contract between {sales_contract_attributes[i]['seller_name']} and {sales_contract_attributes[i]['customer_name']}?",
                "answer": f"{sales_contract_attributes[i]['notify_ddl_days']}."
                },
                {
                "question": f"What was the duration of the cooling-off period in days based on the contract between {sales_contract_attributes[i]['seller_name']} and {sales_contract_attributes[i]['customer_name']}?",
                "answer": f"{sales_contract_attributes[i]['cooling_off_days']}."
                },
                {
                "question": f"Which jurisdiction's laws govern the contract between {sales_contract_attributes[i]['seller_name']} and {sales_contract_attributes[i]['customer_name']}?",
                "answer": f"{sales_contract_attributes[i]['governing_law']}."
                }  
            ]
            }
            )
        return sales_contract_attributes, QA_sales_contract

def main():
    sales_qa_gen = SalesQAGen()
    _, QA_sales_contract = sales_qa_gen.sales_contract_QAgen(entity_pairs)

    QA_sales_contract_list = convert_data(QA_sales_contract)

    with open("data/sample_data_2.json", "w", encoding="utf-8") as json_file:
         json.dump(QA_sales_contract_list, json_file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
    
