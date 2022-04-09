#!/usr/bin/env python
from dataclasses import astuple
from datetime import datetime
import betterproto
from utc_bot import UTCBot, start_bot
import proto.utc_bot as pb

import asyncio

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.stats import norm

option_strikes = [90, 95, 100, 105, 110]
risk_bounds = np.array([2000, 5000, 500000, 5000])
greed = 1
step = 0

vol_list = []

class NoisyNuts(UTCBot):

    async def handle_round_started(self):
        """
        This function is called when the round is started. You should do your setup here, and
        start any tasks that should be running for the rest of the round.
        """
         # Stores the current day (starting from 0 and ending at 5). This is a floating point number,
        # meaning that it includes information about partial days
        self.current_day = 0
        self.step_count = 0
        self.total_risk = 0

        # Stores the current value of the underlying asset
        self.underlying_prices = [100]
        self.underlying_day_price = 100

        # This variable will be a map from asset names to positions. We start out by initializing it
        # to zero for every asset.
        self.positions = {}

        self.positions["UC"] = 0
        self.best_books = {}
        for strike in option_strikes:
            for flag in ["C", "P"]:
                asset_name = f"UC{strike}{flag}"
                self.positions[asset_name] = 0
                self.best_books[asset_name] = None
        
        self.bid_order_id = []
        self.ask_order_id = []

    def getAsset(self, asset_name):
        sub_str = asset_name[2:-1]
        flag = asset_name[-1]
        return int(sub_str), flag

    def compute_vol_estimate(self) -> float:
        prices = np.array(self.underlying_prices)
        
        if len(prices) < 2:
            return 0.08*(252**0.5)*100
        vol = np.std((prices[1:] - prices[:-1])/prices[:-1])*(252**0.5)
        vol_list.append(vol)
        return vol * self.underlying_prices[-1]
        
        #time_expiry = (26 - self.current_day) / 26
        #premium = (self.best_books["UC100P"][0] + self.best_books["UC100P"][1]) / 2
        #iv = iv_put(self.underlying_prices[-1], 100, time_expiry, 0, premium)
        #return np.std(prices)*(252**0.5)
        #return iv

    def compute_options_price(
        self,
        flag: str,
        underlying_px: float,
        strike_px: float,
        time_to_expiry: float,
        volatility: float,
    ) -> float:
        """
        This function should compute the price of an option given the provided parameters.
        """
        assert flag in ['P', 'C']
        if flag == 'P':
            theo_price = bs_put(underlying_px, strike_px, time_to_expiry, 0, volatility)
        else:
            theo_price = bs_call(underlying_px, strike_px, time_to_expiry, 0, volatility)
        return theo_price

    def single_risk(self, strike, flag, time_expiry, vol):
        """
        Return risk for single asset
        """
        # choose to ignore other greeks :D
        if flag == 'P':
            risk = delta_put(self.underlying_prices[-1], strike, time_expiry, vol)
        else:
            risk = delta_call(self.underlying_prices[-1], strike, time_expiry, vol)
        return risk

    def get_risk(self):
        """
        Check our own greeks to guage whether or not to continue.
        """
        self.total_risk = 0
        for strike in option_strikes:
            for flag in ["C", "P"]:
                asset_name = f"UC{strike}{flag}"
                option_price = (self.best_books[asset_name][0] + self.best_books[asset_name][1]) / 2
                time_expiry = (26 - self.current_day) / 26
                risk = self.single_risk(strike, flag, time_expiry, option_price) * self.positions[f'UC{strike}{flag}']
                self.total_risk += risk
        return self.total_risk

    def check_risk(self):
        # we need to clear risk periodically?
        val = self.total_risk / risk_bounds[0]
        if abs(val) > 1:
            val = val / abs(val)
        assert abs(val) <= 1, print('check risk failure')
        return val/greed

    def sign_func(self, x):
        if x >= 0:
            return 1
        return -1

    def weight_risk(self, risk):
        """calculate risk functions for weight"""
        lin_func = lambda r: 7.5*r + 7.5
        quad_func = lambda r: 7.5*r*r*self.sign_func(r) + 7.5
        root_func = lambda r: 7.5*(abs(r)**0.5)*self.sign_func(r) + 7.5
        pos1 = max(int(round(root_func(risk))), 15)
        pos2 = max(int(round(root_func(-risk))), 15)
        return pos1, pos2

    async def update_options_quotes(self):
        """
        This function will update the quotes that the bot has currently put into the market.
        """
        if self.current_day < 0:
            return
        if len(self.underlying_prices) < 2:
            return
        if self.step_count % 5 != 0:
            return

        bid_requests = []
        ask_requests = []
        cancel_requests = []

        # BSM prices
        use_prices = {}
        vol = self.compute_vol_estimate()
        for strike in option_strikes:
            for flag in ["C", "P"]:
                asset_name = f"UC{strike}{flag}"
                expiry_time = (26 - self.current_day)/252
                fair_price = self.compute_options_price(flag, self.underlying_prices[-1], strike, expiry_time, vol)
                use_prices[asset_name] = [round(fair_price, 1) - 0.2, round(fair_price, 1) - 0.2] # bids then asks
                if self.best_books[asset_name] != None:
                    use_prices[asset_name][0] = min(use_prices[asset_name][0], self.best_books[asset_name][0] + 0.1)
                    use_prices[asset_name][1] = max(use_prices[asset_name][1], self.best_books[asset_name][1] - 0.1)
                """
                if self.positions[asset_name] > 50:
                    use_prices[asset_name][1] = self.best_books[asset_name][1] - 0.3
                elif self.positions[asset_name] < -50:
                    use_prices[asset_name][0] = self.best_books[asset_name][0] + 0.2
                """


        # cancel all existing orders  
        for order_id in self.bid_order_id:
            cancel_requests.append(
                self.cancel_order(
                    order_id
                )
            )
        """
        for order_id in self.ask_order_id:
            cancel_requests.append(
                self.cancel_order(
                    order_id
                )
            )
        """
        if len(cancel_requests) > 0:
            cancel_responses = await asyncio.gather(*cancel_requests)
        
        # new orders - first check risk for weighting
        risk = self.check_risk()
        pos = self.weight_risk(risk)
        bid_weights = {"P" : pos[0], "C" : pos[1]}
        ask_weights = {"P" : pos[1], "C" : pos[0]}

        # TODO: add adjusting values based on time from data analysis
        for strike in option_strikes:
            for flag in ["C", "P"]:
                asset_name = f"UC{strike}{flag}"
                # comment out for sig
                #if self.positions[asset_name] > 50:
                #    bid_weights[flag] = 1
                #elif self.positions[asset_name] < -50:
                #    ask_weights[flag] = 1
                if bid_weights[flag] > 0:
                    bid_requests.append(
                        self.place_order(
                            asset_name,
                            pb.OrderSpecType.LIMIT,
                            pb.OrderSpecSide.BID,
                            bid_weights[flag],
                            use_prices[asset_name][0],
                        )
                    )
                if ask_weights[flag] > 0:
                    ask_requests.append(
                        self.place_order(
                            asset_name,
                            pb.OrderSpecType.LIMIT,
                            pb.OrderSpecSide.ASK,
                            ask_weights[flag],
                            #use_prices[asset_name][1],
                            self.best_books[asset_name][1] - 0.1, # 0.2 subtract might be better? imc
                        )
                    )

        # optimization trick -- use asyncio.gather to send a group of requests at the same time
        # instead of sending them one-by-one
        bid_responses = await asyncio.gather(*bid_requests)
        ask_responses = await asyncio.gather(*ask_requests)

        self.bid_order_id = []
        self.ask_order_id = []
        for resp in bid_responses:
            assert resp.ok, resp.message
            self.bid_order_id.append(resp.order_id)
        for resp in ask_responses:
            assert resp.ok, resp.message
            self.ask_order_id.append(resp.order_id)


    async def handle_exchange_update(self, update: pb.FeedMessage):
        kind, _ = betterproto.which_one_of(update, "msg")

        if kind == "pnl_msg":
            # When you hear from the exchange about your PnL, print it out
            print("My PnL:", update.pnl_msg.m2m_pnl)
            print("My Positions:", self.positions)
            print("My Risk:", self.total_risk)

        elif kind == "fill_msg":
            # When you hear about a fill you had, update your positions
            fill_msg = update.fill_msg

            if fill_msg.order_side == pb.FillMessageSide.BUY:
                self.positions[fill_msg.asset] += update.fill_msg.filled_qty
            else:
                self.positions[fill_msg.asset] -= update.fill_msg.filled_qty

            self.total_risk = self.get_risk()

        elif kind == "market_snapshot_msg":
            # When we receive a snapshot of what's going on in the market, update our information
            # about the underlying price.
            self.step_count += 1
            step = self.step_count
            book = update.market_snapshot_msg.books["UC"]

            # Compute the mid price of the market and store it
            if len(book.bids) > 0 and len(book.asks) > 0:
                self.underlying_prices.append(
                    (float(book.bids[0].px) + float(book.asks[0].px)) / 2
                )
                if len(self.underlying_prices) > 10:
                    self.underlying_prices = self.underlying_prices[1:]
                
            orderbook = update.market_snapshot_msg.books
            for strike in option_strikes:
                for flag in ["P", "C"]:
                    asset_name = f"UC{strike}{flag}"
                    if len(orderbook[asset_name].bids) == 0 or len(orderbook[asset_name].asks) == 0:
                        continue
                    self.best_books[asset_name] = [float(orderbook[asset_name].bids[0].px), float(orderbook[asset_name].asks[0].px)]

            await self.update_options_quotes()

        elif (
            kind == "generic_msg"
            and update.generic_msg.event_type == pb.GenericMessageType.MESSAGE
        ):
            # The platform will regularly send out what day it currently is (starting from day 0 at
            # the start of the case) 
            prev_day = self.current_day
            self.current_day = float(update.generic_msg.message)
            if prev_day != self.current_day:
                self.underlying_day_price = self.underlying_prices[-1]
        elif kind == "request_failed_msg":
            # check for request denied
            pass


# Greek formulas self import

def d1(S,K,T,r,sigma):
    return(np.log(S/K)+(r+sigma**2/2.)*T)/(sigma*np.sqrt(T))
def d2(S,K,T,r,sigma):
    return d1(S,K,T,r,sigma)-sigma*np.sqrt(T)

# Implied Volatility:

iters = 20

def iv_call(S,K,T,r,C):
    return max(0, fsolve((lambda sigma: np.abs(bs_call(S,K,T,r,sigma) - C)), [1], maxfev = iters)[0])
                      
def iv_put(S,K,T,r,P):
    return max(0, fsolve((lambda sigma: np.abs(bs_put(S,K,T,r,sigma) - P)), [1], maxfev = iters)[0])

def bs_call(S,K,T,r,sigma):
    return S*norm.cdf(d1(S,K,T,r,sigma))-K*np.exp(-r*T)*norm.cdf(d2(S,K,T,r,sigma))
def bs_put(S,K,T,r,sigma):
    return K*np.exp(-r*T)-S+bs_call(S,K,T,r,sigma)

def delta_call(S,K,T,C):
    sigma = iv_call(S,K,T,0,C)
    return 100 * norm.cdf(d1(S,K,T,0,sigma))

def gamma_call(S,K,T,C):
    sigma = iv_call(S,K,T,0,C)
    return 100 * norm.pdf(d1(S,K,T,0,sigma))/(S * sigma * np.sqrt(T))

def vega_call(S,K,T,C):
    sigma = iv_call(S,K,T,0,C)
    return 100 * norm.pdf(d1(S,K,T,0,sigma)) * S * np.sqrt(T)

def theta_call(S,K,T,C):
    sigma = iv_call(S,K,T,0,C)
    return 100 * S * norm.pdf(d1(S,K,T,0,sigma)) * sigma/(2 * np.sqrt(T))

def delta_put(S,K,T,C):
    sigma = iv_put(S,K,T,0,C)
    return 100 * (norm.cdf(d1(S,K,T,0,sigma)) - 1)

def gamma_put(S,K,T,C):
    sigma = iv_put(S,K,T,0,C)
    return 100 * norm.pdf(d1(S,K,T,0,sigma))/(S * sigma * np.sqrt(T))

def vega_put(S,K,T,C):
    sigma = iv_put(S,K,T,0,C)
    return 100 * norm.pdf(d1(S,K,T,0,sigma)) * S * np.sqrt(T)

def theta_put(S,K,T,C):
    sigma = iv_put(S,K,T,0,C)
    return 100 * S * norm.pdf(d1(S,K,T,0,sigma)) * sigma/(2 * np.sqrt(T))


# run
if __name__ == "__main__":
    start_bot(NoisyNuts)
    print(vol_list)
    print(step)