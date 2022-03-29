#!/usr/bin/env python

from utc_bot import UTCBot, start_bot
import proto.utc_bot as pb
import betterproto
import numpy as np
import asyncio

CONTRACTS = ["LBSJ","LBSM", "LBSQ", "LBSV", "LBSZ"]
EXPIRY_PRICE = { #2016-2021
    "LBSJ": [221.23,210.73,185.91,220.15,219.58,213.2],
    "LBSM": [207.55,209.6,280.91,281.31,255.24,208.49],
    "LBSQ": [246.63,276.45,260.07,227.84,242.59,258.71],
    "LBSV": [262.2,296.24,244.98,228.33,242.1,270.32],
    "LBSZ": [282.58,260.49,265.63,280.89,311.71,330.02]
}


""" some ideas:
    fix theo using some loose estimate from rain data?

    if nobody is taking my market, that means they think theo is inside my spread. tighten the spread to try to fill orders
    if people take my ask, they think theo > spread and we are short. increase our theo, and bid a little higher to neutralize position (increase order size)
    similarly, if people take my bid, they think theo < spread and we are long. decrease our theo, and ask a little lower to neutralize position.
    we can get this from a cache that keeps the last 10? transactions. like a queue
    

    we should always try to keep neutral profile
"""
class Case1ExampleBot(UTCBot):
    '''
    An example bot for Case 1 of the 2022 UChicago Trading Competition. We recommend that you start
    by reading through this bot and understanding how it works. Then, make a copy of this file and
    start trying to write your own bot!
    '''

    async def handle_round_started(self):
        '''
        This function is called when the round is started. You should do your setup here, and
        start any tasks that should be running for the rest of the round.
        '''
        self.rain = []
        self.fairs = {}
        self.order_book = {}
        self.pos = {}
        self.order_ids = {}
        self.trading_rounds = 0
        self.history = {} # who is taking our bid/ask 
        self.state = {} # sell, normal, buy: inventory control state
        self.bid_size = {}
        self.ask_size = {}
        self.bid_spread = {}
        self.ask_spread = {}
        self.pnl = 0

        self.MIN_SPREAD = 5
        for month in CONTRACTS:
            self.order_ids[month+' bid'] = ''
            self.order_ids[month+' ask'] = ''

            self.fairs[month] = 230

            self.history[month] = [0.1] * 5 # queue storing the last 5 timesteps

            self.order_book[month] = {
                'Best Bid':{'Price':0,'Quantity':0},
                'Best Ask':{'Price':0,'Quantity':0}}
            
            self.state[month] = "normal"

            self.bid_size[month] = 10
            self.ask_size[month] = 10
            self.bid_spread[month] = 60
            self.ask_spread[month] = 60

            self.pos[month] = 0

        asyncio.create_task(self.update_quotes())

    def update_fairs(self):
        '''
        You should implement this function to update the fair value of each asset as the
        round progresses.
        '''
        print(self.rain)
        print(self.state)
        # print(self.order_book)
        pass
        
    def market_trend(self, id):
        # looks at self.history for each stock
        # if trending up (80% ask) or trending low (80% bid) then adjust quotes
        hist = self.history[id]
        if np.count_nonzero(hist) < 1:
            return "loose"
        if np.sum(hist) >= 3:
            return "high"
        if np.sum(hist) <= -3:
            return "low"
        if np.count_nonzero(hist) == 5:
            return "tight"
        return ""

    async def update_quotes(self):
        '''
        This function updates the quotes at each time step. In this sample implementation we 
        are always quoting symetrically about our predicted fair prices, without consideration
        for our current positions. We don't reccomend that you do this for the actual competition.
        '''

        unit = 0.3
        while True:

            self.update_fairs()

            for contract in CONTRACTS:

                ## synthesize market trends to adjust spread
                trend = self.market_trend(contract)
                if trend == "high":
                    self.fairs[contract] += 0.3
                    self.bid_spread[contract] += unit 
                    self.bid_size[contract] += 1
                elif trend == "low":
                    self.fairs[contract] -= 0.3 # something
                    self.ask_spread[contract] += unit
                    self.ask_size[contract] += 1
                elif trend == "loose":
                    self.bid_spread[contract] -= unit
                    self.ask_spread[contract] -= unit
                elif trend == "tight":
                    self.bid_spread[contract] += unit
                    self.ask_spread[contract] += unit

                ## check current position and sets state. changes bid, asks, but not the theo
                if self.pos[contract] >= 50:
                    # ask the bot to dump contracts
                    self.state[contract] = "sell"

                elif self.pos[contract] <= -50:
                    # increase bid price, increase ask
                    self.state[contract] = "buy"

                if self.state[contract] == "sell":
                    bid_size, ask_size = self.bid_size[contract]-3, self.ask_size[contract]+3
                    self.bid_spread[contract] -= 0.2
                    self.ask_spread[contract] -= 0.2 # lower ask to sell position   

                    if self.pos[contract] <= 20:
                        self.state[contract] = "normal"  
                        
                elif self.state[contract] == "buy":
                    bid_size, ask_size = self.bid_size[contract]+3, self.ask_size[contract]-3
                    self.bid_spread[contract] -= 1 # increase bid to buy back
                    self.ask_spread[contract] += 1

                    if self.pos[contract] >= -20:
                        self.state[contract] = "normal"

                else:
                    bid_size, ask_size = self.bid_size[contract], self.ask_size[contract]

                ## post our spread
                bid_response = await self.modify_order(
                    self.order_ids[contract+' bid'],
                    contract,
                    pb.OrderSpecType.LIMIT,
                    pb.OrderSpecSide.BID,
                    bid_size,
                    round(self.fairs[contract]-self.bid_spread[contract],2))

                ask_response = await self.modify_order(
                    self.order_ids[contract+' ask'],
                    contract,
                    pb.OrderSpecType.LIMIT,
                    pb.OrderSpecSide.ASK,
                    ask_size,
                    round(self.fairs[contract]+self.ask_spread[contract],2))

                self.history[contract].pop(0)
                self.history[contract].append(0)
                assert bid_response.ok
                self.order_ids[contract+' bid'] = bid_response.order_id  
                    
                assert ask_response.ok
                self.order_ids[contract+' ask'] = ask_response.order_id  

                print(f'{round(self.fairs[contract]-self.bid_spread[contract],2)}@{round(self.fairs[contract]+self.ask_spread[contract],2)},{bid_size},{ask_size}')
            
            await asyncio.sleep(1)

    async def handle_exchange_update(self, update: pb.FeedMessage):
        '''
        This function receives messages from the exchange. You are encouraged to read through
        the documentation for the exachange to understand what types of messages you may receive
        from the exchange and how they may be useful to you.
        
        Note that monthly rainfall predictions are sent through Generic Message.
        '''
        kind, _ = betterproto.which_one_of(update, "msg")

        if kind == "pnl_msg":
            print('Realized pnl:', update.pnl_msg.realized_pnl)
            print("M2M pnl:", update.pnl_msg.m2m_pnl)
            print(self.pos)

        elif kind == "market_snapshot_msg":
        # Updates your record of the Best Bids and Best Asks in the market
            for contract in CONTRACTS:
                book = update.market_snapshot_msg.books[contract]
                if len(book.bids) != 0:
                    best_bid = book.bids[0]
                    self.order_book[contract]['Best Bid']['Price'] = float(best_bid.px)
                    self.order_book[contract]['Best Bid']['Quantity'] = best_bid.qty

                if len(book.asks) != 0:
                    best_ask = book.asks[0]
                    self.order_book[contract]['Best Ask']['Price'] = float(best_ask.px)
                    self.order_book[contract]['Best Ask']['Quantity'] = best_ask.qty
        
        elif kind == "fill_msg":
            # When you hear about a fill you had, update your positions
            fill_msg = update.fill_msg

            if fill_msg.order_side == pb.FillMessageSide.BUY:
                self.pos[fill_msg.asset] += update.fill_msg.filled_qty
                self.history[fill_msg.asset][-1] = 1
            else:
                self.pos[fill_msg.asset] -= update.fill_msg.filled_qty
                self.history[fill_msg.asset][-1] = -1

        elif kind == "generic_msg":
            # Saves the predicted rainfall
            try:
                pred = float(update.generic_msg.message)
                print("New rain forecst:", pred)
                self.rain.append(pred)
            # Prints the Risk Limit message
            except ValueError:
                msg = update.generic_msg.message
                if "ROUND_ENDED" in msg:
                   print(self.pos)
                
                liquidate = 0
                for contract in CONTRACTS:
                    # get expiry price
                    year_idx = self.year - 2016
                    price = (EXPIRY_PRICE[contract])[year_idx]
                    liquidate += self.pos[contract] * price
                
                print("Final pnl:", self.pnl + liquidate)
                print(update.generic_msg.message)


if __name__ == "__main__":
    start_bot(Case1ExampleBot)