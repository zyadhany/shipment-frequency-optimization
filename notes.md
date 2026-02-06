- ABC Clacifcatiions

JIT (Just in Time)



orders.csv
current schecme:
order_id,dest_location_id,qty,required_date,priority

optionals: sku_id, priority, earliest_ship_date



transport_costs.csv
current schecme:
lane_id,origin_location_id,dest_location_id,lead_time,unit_cost,fixed_cost


capacity.csv
current schecme:
origin_location_id,truck_qty,capacity_of_truck


plan.csv
order_id,ship_date,lane_id,allocated_qty,transport_cost

