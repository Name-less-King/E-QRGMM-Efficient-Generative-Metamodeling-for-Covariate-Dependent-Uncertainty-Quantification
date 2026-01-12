import numpy as np
import numba as nb

@nb.njit
def simulate_inventory_core(s, S, initial, periods, h, f, c, demands, lead_times):
    max_lead = np.max(lead_times)
    arrivals_array = np.zeros(periods + max_lead + 2)  # +2: safe for t + lead + 1

    stock_on_hand = initial
    backorders = 0
    total_cost = 0.0
    outstanding_total = 0
    lead_time_index = 0

    for t in range(periods):
        # Step 1: Receive arrivals
        arrivals = arrivals_array[t]
        stock_on_hand += arrivals
        outstanding_total -= arrivals

        # Step 2: Fill backorders
        if backorders > 0:
            fill = min(stock_on_hand, backorders)
            backorders -= fill
            stock_on_hand -= fill

        # Step 3: Serve today's demand
        demand = demands[t]
        if demand <= stock_on_hand:
            stock_on_hand -= demand
        else:
            backorders += demand - stock_on_hand
            stock_on_hand = 0

        # Step 4: Inventory position
        inventory_position = stock_on_hand - backorders + outstanding_total

        # Step 5: Place order if needed
        if inventory_position < s:
            lead_time = lead_times[lead_time_index]
            lead_time_index += 1

            order_qty = S - inventory_position
            arrival_period = t + lead_time + 1
            arrivals_array[arrival_period] += order_qty
            outstanding_total += order_qty
            total_cost += f + c * order_qty

        # Step 6: Holding cost
        total_cost += h * stock_on_hand

    return total_cost / periods
