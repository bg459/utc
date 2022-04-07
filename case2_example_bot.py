#!/usr/bin/env python

from dataclasses import astuple
from datetime import datetime
import betterproto
from utc_bot import UTCBot, start_bot
import proto.utc_bot as pb

import asyncio

import Greeks_Formulas as gf
import numpy as np


option_strikes = [90, 95, 100, 105, 110]
risk_bounds = [2000, 5000, 1000000, 5000]

class Case2ExampleBot(UTCBot):
    """
    An example bot for Case 2 of the 2021 UChicago Trading Competition. We recommend that you start
    by reading through this bot and understanding how it works. Then, make a copy of this file and
    start trying to write your own bot!
    """

    async def handle_round_started(self):
        """
        This function is called when the round is started. You should do your setup here, and
        start any tasks that should be running for the rest of the round.
        """
         # Stores the current day (starting from 0 and ending at 5). This is a floating point number,
        # meaning that it includes information about partial days
        self.current_day = 0
        self.total_risk = [0, 0, 0, 0]

        # Stores the current value of the underlying asset
        self.underlying_prices = [100]
        self.underlying_day_price = 100

        # This variable will be a map from asset names to positions. We start out by initializing it
        # to zero for every asset.
        self.fair_price = {}
        self.positions = {}

        self.positions["UC"] = 0
        self.best_books = {}
        px_pair = self.compute_end_underlying(0.90)
        for strike in option_strikes:
            for flag in ["C", "P"]:
                asset_name = f"UC{strike}{flag}"
                self.positions[asset_name] = 0
                self.fair_price[asset_name] = [
                    self.compute_options_price(flag, px_pair[0], strike, 21/252, 0.0795),
                    self.compute_options_price(flag, px_pair[1], strike, 21/252, 0.0795),
                ]
                self.best_books[asset_name] = None
        
        self.bid_order_id = []
        self.ask_order_id = []

    def getAsset(self, asset_name):
        sub_str = asset_name[2:-1]
        flag = asset_name[-1]
        return int(sub_str), flag

    def compute_vol_estimate(self) -> float:
        """
        This function is used to provide an estimate of underlying's volatility. Because this is
        an example bot, we just use a placeholder value here. We recommend that you look into
        different ways of finding what the true volatility of the underlying is.
        """
        vol = np.array(self.underlying_prices)
        return np.std(vol)*(252**0.5)

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
            theo_price = gf.bs_put(underlying_px, strike_px, time_to_expiry, 0, volatility)
        else:
            theo_price = gf.bs_call(underlying_px, strike_px, time_to_expiry, 0, volatility)
        return theo_price

    def compute_end_underlying(
        self,
        confidence: float,
    ) -> float:
        """
        This functions estimates a confidence interval for the underlying price at the end of the round.
        """
        return self.underlying_prices[-1] - 2, self.underlying_prices[-1] + 2

    def single_risk(self, strike, flag, time_expiry, vol):
        """
        Return risk for single asset
        """
        risk = [0, 0, 0, 0]
        if flag == 'P':
            risk[0] = gf.delta_put(self.underlying_prices[-1], strike, time_expiry, vol)
            risk[1] = gf.gamma_put(self.underlying_prices[-1], strike, time_expiry, vol)
            risk[2] = gf.vega_put(self.underlying_prices[-1], strike, time_expiry, vol)
            risk[3] = gf.theta_put(self.underlying_prices[-1], strike, time_expiry, vol)
        else:
            risk[0] = gf.delta_call(self.underlying_prices[-1], strike, time_expiry, vol)
            risk[1] = gf.gamma_call(self.underlying_prices[-1], strike, time_expiry, vol)
            risk[2] = gf.vega_call(self.underlying_prices[-1], strike, time_expiry, vol)
            risk[3] = gf.theta_call(self.underlying_prices[-1], strike, time_expiry, vol)
        return np.array(risk)

    def get_risk(self):
        """
        Check our own greeks to guage whether or not to continue.
        """
        total_risk = np.array([0, 0, 0, 0])
        for strike in option_strikes:
            for flag in ["C", "P"]:
                risk = self.single_risk(strike, flag, (26 - self.current_day) / 26, self.compute_vol_estimate()) * self.positions[f'UC{strike}{flag}']
                total_risk = total_risk + risk
        return total_risk

    def check_risk(self, risk, critical):
        risky = False
        for i in range(4):
            if (risk_bounds[i] - abs(risk[i])) / risk_bounds[i] < critical:
                risky = True
        return risky

    async def update_options_quotes(self):
        """
        This function will update the quotes that the bot has currently put into the market.

        In this example bot, the bot won't bother pulling old quotes, and will instead just set new
        quotes at the new theoretical price every time a price update happens. We don't recommend
        that you do this in the actual competition
        """
        bid_requests = []
        ask_requests = []
        cancel_requests = []
        risky = self.check_risk(self.total_risk, 0.3)

        for strike in option_strikes:
            for flag in ["C", "P"]:
                asset_name = f"UC{strike}{flag}"
                px_pair = self.compute_end_underlying(0.90)
                """
                self.fair_price[f"UC{strike}{flag}"] = [
                    self.compute_options_price(flag, px_pair[0], strike, 21/252, self.compute_vol_estimate()),
                    self.compute_options_price(flag, px_pair[1], strike, 21/252, self.compute_vol_estimate()),
                ]
                """
                self.fair_price[f"UC{strike}{flag}"] = [
                    self.compute_options_price(flag, self.underlying_prices[-1], strike, (26 - self.current_day)/252, self.compute_vol_estimate()),
                    self.compute_options_price(flag, self.underlying_prices[-1], strike, (26 - self.current_day)/252, self.compute_vol_estimate()),
                ]
                
        for order_id in self.bid_order_id:
            cancel_requests.append(
                self.cancel_order(
                    order_id
                )
            )
        for order_id in self.ask_order_id:
            cancel_requests.append(
                self.cancel_order(
                    order_id
                )
            )
        if len(cancel_requests) > 0:
            cancel_responses = await asyncio.gather(*cancel_requests)
        
        # new orders
        # TODO: add adjusting values based on time from data analysis
        risky = False # bro idk how to control risk even if it is risky :(
        if not risky:
            for strike in option_strikes:
                for flag in ["C", "P"]:
                    asset_name = f"UC{strike}{flag}"
                    use_price = min(self.best_books[asset_name][0] + 0.1, self.fair_price[asset_name][0] - 0.2)
                    bid_requests.append(
                        self.place_order(
                            asset_name,
                            pb.OrderSpecType.LIMIT,
                            pb.OrderSpecSide.BID,
                            9,  # TODO: How should this quantity be chosen?
                            round(use_price, 1),
                        )
                    )
            for strike in option_strikes:
                for flag in ["C", "P"]:
                    asset_name = f"UC{strike}{flag}"
                    use_price = max(self.best_books[asset_name][1] - 0.1, self.fair_price[asset_name][1])
                    ask_requests.append(
                        self.place_order(
                            asset_name,
                            pb.OrderSpecType.LIMIT,
                            pb.OrderSpecSide.ASK,
                            15,
                            round(use_price, 1),
                        )
                    )
        else:
            # buying puts
            for strike in option_strikes:
                for flag in ["C"]:
                    asset_name = f"UC{strike}{flag}"
                    use_price = min(self.best_books[asset_name][0] + 0.1, self.fair_price[asset_name][0] - 0.2)
                    bid_requests.append(
                        self.place_order(
                            asset_name,
                            pb.OrderSpecType.LIMIT,
                            pb.OrderSpecSide.BID,
                            9,  # TODO: How should this quantity be chosen?
                            round(use_price, 1),
                        )
                    )
            for strike in option_strikes:
                for flag in ["P"]:
                    asset_name = f"UC{strike}{flag}"
                    ask_requests.append(
                        self.place_order(
                            asset_name,
                            pb.OrderSpecType.LIMIT,
                            pb.OrderSpecSide.ASK,
                            15,
                            round(self.best_books[asset_name][1] - 0.1, 1),
                        )
                    )

        # optimization trick -- use asyncio.gather to send a group of requests at the same time
        # instead of sending them one-by-one
        bid_responses = await asyncio.gather(*bid_requests)
        ask_responses = await asyncio.gather(*ask_requests)

        self.bid_order_id = []
        self.ask_order_id = []
        for resp in bid_responses:
            assert resp.ok
            self.bid_order_id.append(resp.order_id)
        for resp in ask_responses:
            assert resp.ok
            self.ask_order_id.append(resp.order_id)


    async def handle_exchange_update(self, update: pb.FeedMessage):
        kind, _ = betterproto.which_one_of(update, "msg")

        if kind == "pnl_msg":
            # When you hear from the exchange about your PnL, print it out
            print("My PnL:", update.pnl_msg.m2m_pnl)
            print("My Positions:", self.positions)
            print("My Risk:", self.total_risk)
            print("Risky:", self.check_risk(self.total_risk, 0.3))

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


if __name__ == "__main__":
    start_bot(Case2ExampleBot)