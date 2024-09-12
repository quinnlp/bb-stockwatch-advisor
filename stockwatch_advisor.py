#!/usr/bin/env python3

import argparse
import scipy
import statistics
import tabulate
import yaml

# This script is a tool to advise participants in the Big Brother Stockwatch (https://realitystockwatch.com) run by Rob Has a Podcast (https://robhasawebsite.com/).
# This script identifies the trade actions given a net-worth that maximizes expected return based on the average projection of the Big Brother houseguests accoring to the current stockwatch ratings.
# This script uses scipy.optimize.linprog (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html) to minimize c @ x such that A_eq @ x == b_eq and 0 <= x where
# c := [-projection_1, -projection_2, ..., -projection_n, -0.01]
#     projection_i := average projection of houseguest i
#     n := number_of_houseguests
#     note: negative elements since linprog can only minimize
#     note: -0.01 is appended since you can hold your money
# x := [stock_1, stock_2, ..., stock_n, holdings]
#     stock_i := the number of stocks the script advises you to purchase of houseguest i
#     holdings := the amount of cents the script advises you to hold
#     n := number_of_houseguests
#     note: stock_i is an integer since you must cannot purchase fractions of a stock
#     note: holdings is an integer since you cannot hold fractions of a cent
# A_eq := [[cost_1, cost_2, ..., cost_n, 0.01]]
#     cost_i := the current cost of purchasing a stock of houseguest i
#     n := number_of_houseguests
#     note: 0.01 is appended since you can hold your money
# b_eq := [net_worth]
#     net_worth := the amount of money you can use to purchase stocks or hold

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", help="yaml file containing the stockwatch state")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="enable verbose output")
    args = parser.parse_args()

    with open(args.filepath) as file:
        yamlfile = yaml.safe_load(file)
        houseguests = yamlfile["houseguests"]

        c = []
        A_eq = [[]]
        b_eq = [yamlfile["net_worth"]]
        bounds = []
        integrality = []
        for houseguest in houseguests:
            c.append(-statistics.mean(houseguest["projections"]))
            A_eq[0].append(houseguest["cost"])
            bounds.append((0, None)) # you cannot purchase fewer than 0 stocks and there is no upper bound
            integrality.append(1) # stock variables must be integers

        c.append(-0.01)
        A_eq[0].append(0.01)
        bounds.append((0, None)) # you cannot hold fewer than 0 cents and there is no upper bound
        integrality.append(1) # holdings variable must be an integer

        res = scipy.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs", integrality=integrality) # highs method must be used to utilize integrality

        if args.verbose:
            table = []
            for houseguest, cost, projection in zip(houseguests, A_eq[0], c):
                table.append([houseguest["name"], cost, houseguest["projections"], -projection, (-projection - cost) / cost])
            print(f"{tabulate.tabulate(table, headers=['Name', 'Cost ($)', 'Projections ($)', 'Average Projections ($)', 'Expected Change (%)'])}\n")

        print(f"Net worth: ${b_eq[0]}\n")

        print("Advice")
        print("------")
        for houseguest, stock, cost, projection in zip(houseguests, res.x, A_eq[0], c):
            if stock > 0 or args.verbose:
                print(f"{houseguest['name']}: {round(stock)} (${round(stock) * cost} -> ${round(stock) * -projection})")
        if res.x[-1] > 0 or args.verbose:
            print(f"Holdings: {res.x[-1]} (${res.x[-1] / 100} -> ${res.x[-1] / 100})")
        print("------\n")

        print(f"Expected Value: ${-res.fun} ({(-res.fun - b_eq[0]) / b_eq[0]}%)\n")

        print(f"Eviction Probability: {yamlfile['evictions'] / len(houseguests)}")

if __name__ == "__main__":
    main()
