import json
import random

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
job_pisition = data_input["job_pisition"]
employer_benefit_plans = data_input["employer_benefit_plans"]
salary_frequency = data_input["salary_frequency"]

sales_edge_note = ['A_B', 'A_C', 'A_C2', 'A_C3', 'B_D', 'B_D2', 'B_D3', 'E1_F1', 'E2_F2', 'E3_F3']
employment_edge_note = ['A_m', 'A_n', 'A_n2', 'A_n3', 'B_p', 'B_p2', 'B_p3', 'E1_q1', 'E2_q2', 'E3_q3']

# Generate entity attributes
def name_gen():
    # Generate parties for sales contracts
    sales_edge_note_split = [item.split('_') for item in sales_edge_note]
    unique_company_list = list({item for sublist in sales_edge_note_split for item in sublist})
    company_name_dict = {}
    company_address_dict = {}
    for i in range(len(unique_company_list)):
        company_name_dict.update({unique_company_list[i]: content_gen.company_name_gen(short_name=True)})
        company_address_dict.update({unique_company_list[i]: content_gen.address_gen()})
    seller_name_list, customer_name_list, seller_address_list, customer_address_list = [], [], [], []
    for edge in sales_edge_note_split:
        seller_idx = random.randint(0, 1)
        seller = edge[seller_idx]
        seller_name_list.append(company_name_dict[seller])
        seller_address_list.append(company_address_dict[seller])
        customer = edge[1] if edge[0] == seller else edge[0]
        customer_name_list.append(company_name_dict[customer])
        customer_address_list.append(company_address_dict[customer])


    # Generate parties for employment contracts
    employment_edge_note_split = [item.split('_') for item in employment_edge_note]    
    employee_name_dict = {}
    employee_address_dict = {}
    for i in range(len(employment_edge_note_split)):
        employee_name_dict.update({employment_edge_note_split[i][-1]: content_gen.name_gen()})
        employee_address_dict.update({employment_edge_note_split[i][-1]: content_gen.address_gen()})
    employer_name_list, employer_address_list, employee_name_list, employee_address_list = [], [], [], []
    for edge in employment_edge_note_split:
        employer = edge[0]
        employer_name_list.append(company_name_dict[employer])
        employer_address_list.append(company_address_dict[employer])
        employee = edge[1]
        employee_name_list.append(employee_name_dict[employee])
        employee_address_list.append(employee_address_dict[employee])

    return [seller_name_list, seller_address_list, customer_name_list, customer_address_list, employer_name_list, employer_address_list, employee_name_list, employee_address_list]


class SalesQAGen():
    def __init__(self, combined_name_list):
        self.edge_note = sales_edge_note
        self.num_sales_contract = len(self.edge_note)
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
        self.combined_name_list = combined_name_list
        
    # Generate sales contract attributes
    def sales_contract_QAgen(self):
        seller_name_list = self.combined_name_list[0]
        seller_address_list = self.combined_name_list[1]
        customer_name_list = self.combined_name_list[2]
        customer_address_list = self.combined_name_list[3]
        sales_contract_attributes = []
        QA_sales_contract = {}

        for i in range(self.num_sales_contract):
            sales_contract_attributes.append(
                {
                    'contract_index': f"sales_contract_{i}",
                    'edge_note': self.edge_note[i],
                    'effective_date': content_gen.date_gen(self.start_year, self.end_year),
                    'seller_name': seller_name_list[i],
                    'seller_address': seller_address_list[i],
                    'customer_name': customer_name_list[i],
                    'customer_address': customer_address_list[i],
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

class EmploymentQAGen():
    def __init__(self, combined_name_list):
        self.edge_note = employment_edge_note
        self.num_employment_contract = len(self.edge_note)
        self.start_year = 1980
        self.end_year = 2019
        self.min_employment_months = 6
        self.max_employment_months = 48
        self.early_work_start_hour = 6
        self.late_work_start_hour = 10
        self.early_work_end_hour = 14
        self.late_work_end_hour = 18
        self.min_hourly_pay = 10
        self.max_hourly_pay = 100
        self.min_holiday_days = 10
        self.max_holiday_days = 30
        self.min_confidentiality_months = 3
        self.max_confidentiality_months = 12
        self.min_sick_pay_days = 7
        self.max_sick_pay_days = 14
        self.min_termination_notice_weeks = 2
        self.max_termination_notice_weeks = 12
        self.min_non_compete_months = 6
        self.max_non_compete_months = 24
        self.min_change_notice_weeks = 2
        self.max_change_notice_weeks = 8
        self.combined_name_list = combined_name_list
    
    # Generate employment contract attributes
    def employment_contract_QAgen(self):
        employer_name_list = self.combined_name_list[4]
        employer_address_list = self.combined_name_list[5]
        employee_name_list = self.combined_name_list[6]
        employee_address_list = self.combined_name_list[7]
        employment_contract_attributes = []
        QA_employment_contract = {}
        for i in range(self.num_employment_contract):
            employment_contract_attributes.append(
                {
                    'contract_index': f"employment_contract_{i}",
                    'edge_note': self.edge_note[i],
                    'employer_name': employer_name_list[i],
                    'employer_principal_address': employer_address_list[i],
                    'employee_name': employee_name_list[i],
                    'employee_address': employee_address_list[i],
                    'employment_start_date': content_gen.date_gen(self.start_year, self.end_year),
                    'employment_length': random.randint(self.min_employment_months, self.max_employment_months),
                    'job_position': job_pisition[i],
                    'place_of_work': content_gen.address_gen(),
                    'work_start_hour': random.randint(self.early_work_start_hour, self.late_work_start_hour),
                    'work_end_hour': random.randint(self.early_work_end_hour, self.late_work_end_hour),
                    'hourly_pay': random.randint(self.min_hourly_pay, self.max_hourly_pay),
                    'salary_frequency': random.choice(salary_frequency),
                    'employer_benefit_plans': random.choice(employer_benefit_plans),
                    'holiday_days': random.randint(self.min_holiday_days, self.max_holiday_days),
                    'confidentiality_months': random.randint(self.min_confidentiality_months, self.max_confidentiality_months),
                    'sick_pay_days': random.randint(self.min_sick_pay_days, self.max_sick_pay_days),
                    'termination_notice_weeks': random.randint(self.min_termination_notice_weeks, self.max_termination_notice_weeks),
                    'non_compete_months': random.randint(self.min_non_compete_months, self.max_non_compete_months),
                    'change_notice_weeks': random.randint(self.min_change_notice_weeks, self.max_change_notice_weeks),
                    'governing_law': random.choice(governing_law),
                }
            )
        
            QA_employment_contract.update(
            {
            employment_contract_attributes[i]['edge_note']:
            [
                {
                "question": f"What was the name of the employer in the employment contract with {employment_contract_attributes[i]['employee_name']}, which started from {employment_contract_attributes[i]['employment_start_date']}?",
                "answer": f"{employment_contract_attributes[i]['employer_name']}."
                },
                {
                "question": f"What was the principal business location of {employment_contract_attributes[i]['employer_name']} based on the contract between {employment_contract_attributes[i]['employer_name']} and {employment_contract_attributes[i]['employee_name']}?",
                "answer": f"{employment_contract_attributes[i]['employer_principal_address']}."
                },
                {
                "question": f"What was the name of the employee in the employment contract with {employment_contract_attributes[i]['employer_name']}, which started from {employment_contract_attributes[i]['employment_start_date']}?",
                "answer": f"{employment_contract_attributes[i]['employee_name']}."
                },
                {
                "question": f"What was the address of {employment_contract_attributes[i]['employee_name']} based on the contract between {employment_contract_attributes[i]['employer_name']} and {employment_contract_attributes[i]['employee_name']}?",
                "answer": f"{employment_contract_attributes[i]['employee_address']}."
                },
                {
                "question": f"What was the start date based on the contract between {employment_contract_attributes[i]['employer_name']} and {employment_contract_attributes[i]['employee_name']}?",
                "answer": f"{employment_contract_attributes[i]['employment_start_date']}."
                },
                {
                "question": f"For how many months will the employer employ the employee based on the contract between {employment_contract_attributes[i]['employer_name']} and {employment_contract_attributes[i]['employee_name']}?",
                "answer": f"{employment_contract_attributes[i]['employment_length']} months."
                },
                {
                "question": f"What was the job position based on the contract between {employment_contract_attributes[i]['employer_name']} and {employment_contract_attributes[i]['employee_name']}?",
                "answer": f"{employment_contract_attributes[i]['job_position']}."
                },
                {
                "question": f"What was the work location based on the contract between {employment_contract_attributes[i]['employer_name']} and {employment_contract_attributes[i]['employee_name']}?",
                "answer": f"{employment_contract_attributes[i]['place_of_work']}."
                },
                {
                "question": f"At what hour did the workday start based on the contract between {employment_contract_attributes[i]['employer_name']} and {employment_contract_attributes[i]['employee_name']}?",
                "answer": f"{employment_contract_attributes[i]['work_start_hour']}."
                },
                {
                "question": f"At what hour did the workday finish based on the contract between {employment_contract_attributes[i]['employer_name']} and {employment_contract_attributes[i]['employee_name']}?",
                "answer": f"{employment_contract_attributes[i]['work_end_hour']}."
                },
                {
                "question": f"What was the hourly basic pay in dollars based on the contract between {employment_contract_attributes[i]['employer_name']} and {employment_contract_attributes[i]['employee_name']}?",
                "answer": f"{employment_contract_attributes[i]['hourly_pay']}."
                },
                {
                "question": f"What was the frequency of salary payment based on the contract between {employment_contract_attributes[i]['employer_name']} and {employment_contract_attributes[i]['employee_name']}?",
                "answer": f"{employment_contract_attributes[i]['salary_frequency']}."
                },
                {
                "question": f"What benefit was provided to the employee based on the contract between {employment_contract_attributes[i]['employer_name']} and {employment_contract_attributes[i]['employee_name']}?",
                "answer": f"{employment_contract_attributes[i]['employer_benefit_plans']}."
                },
                {
                "question": f"How many days of paid holiday leave were provided to the employee based on the contract between {employment_contract_attributes[i]['employer_name']} and {employment_contract_attributes[i]['employee_name']}?",
                "answer": f"{employment_contract_attributes[i]['holiday_days']}."
                },
                {
                "question": f"For how many months after the employment ends was the employee prohibited from disclosing any confidential information based on the contract between {employment_contract_attributes[i]['employer_name']} and {employment_contract_attributes[i]['employee_name']}?",
                "answer": f"{employment_contract_attributes[i]['confidentiality_months']}."
                },
                {
                "question": f"What was the number of days the employee was entitled to Paid Sick Leave in each year of employment based on the contract between {employment_contract_attributes[i]['employer_name']} and {employment_contract_attributes[i]['employee_name']}?",
                "answer": f"{employment_contract_attributes[i]['sick_pay_days']}."
                },
                {
                "question": f"How many weeks' written notice of termination must the employee and employer each provide to the other based on the contract between {employment_contract_attributes[i]['employer_name']} and {employment_contract_attributes[i]['employee_name']}?",
                "answer": f"{employment_contract_attributes[i]['termination_notice_weeks']}."
                },
                {
                "question": f"For how many months did the non-compete clause cover based on the contract between {employment_contract_attributes[i]['employer_name']} and {employment_contract_attributes[i]['employee_name']}?",
                "answer": f"{employment_contract_attributes[i]['non_compete_months']}."
                },
                {
                "question": f"How many weeks' written notice must the employer provide before any proposed changes to the terms of employment based on the contract between {employment_contract_attributes[i]['employer_name']} and {employment_contract_attributes[i]['employee_name']}?",
                "answer": f"{employment_contract_attributes[i]['change_notice_weeks']}."
                },
                {
                "question": f"Which jurisdiction's laws govern the contract between {employment_contract_attributes[i]['employer_name']} and {employment_contract_attributes[i]['employee_name']}?",
                "answer": f"{employment_contract_attributes[i]['governing_law']}."
                }
            ]
            }  
            )

        return employment_contract_attributes, QA_employment_contract


def main():
    combined_name_list = name_gen()

    sales_qa_gen = SalesQAGen(combined_name_list)
    _, QA_sales_contract = sales_qa_gen.sales_contract_QAgen()
    employment_qa_gen = EmploymentQAGen(combined_name_list)
    _, QA_employment_contract = employment_qa_gen.employment_contract_QAgen()

    contract_QA = {**QA_sales_contract, **QA_employment_contract}
    contract_QA_list = convert_data(contract_QA)
    with open("data/sample_data_1.json", "w", encoding="utf-8") as json_file:
         json.dump(contract_QA_list, json_file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()

