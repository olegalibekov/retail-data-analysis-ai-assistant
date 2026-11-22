"""Hard spending cap for the demo project.

A Cloud Billing budget only sends notifications. This function receives them
through Pub/Sub and, once the reported cost exceeds the budget, detaches the
project from its billing account. That stops every paid service in the project,
the web demo included, until billing is linked again by hand.

Budget notifications lag behind real usage by hours, so the final bill can end
slightly above the budget.
"""

import base64
import json
import os

import functions_framework
from google.cloud import billing_v1

PROJECT = os.environ["TARGET_PROJECT"]


@functions_framework.cloud_event
def stop_billing(event) -> None:
    notification = json.loads(base64.b64decode(event.data["message"]["data"]))
    cost, budget = notification["costAmount"], notification["budgetAmount"]
    if cost <= budget:
        print(f"Cost {cost} is within the budget {budget}: nothing to do.")
        return

    client = billing_v1.CloudBillingClient()
    name = f"projects/{PROJECT}"
    if not client.get_project_billing_info(name=name).billing_enabled:
        print("Billing is already disabled.")
        return
    client.update_project_billing_info(name=name, project_billing_info={"billing_account_name": ""})
    print(f"Cost {cost} exceeded the budget {budget}: billing disabled for {PROJECT}.")
