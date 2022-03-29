# For subclassing. Takes care of logistical things,
# subclasses should implement algorithm logic

from utc_bot import UTCBot, start_bot
import proto.utc_bot as pb
import betterproto
import sys
import asyncio

CONTRACTS = ["LBSJ","LBSM", "LBSQ", "LBSV", "LBSZ"]
MIN_SPREAD = 20
SPREAD = 2

EXPIRY_PRICE = { #2016-2021
    "LBSJ": [221.23,210.73,185.91,220.15,219.58,213.2],
    "LBSM": [207.55,209.6,280.91,281.31,255.24,208.49],
    "LBSQ": [246.63,276.45,260.07,227.84,242.59,258.71],
    "LBSV": [262.2,296.24,244.98,228.33,242.1,270.32],
    "LBSZ": [282.58,260.49,265.63,280.89,311.71,330.02]
}
import logging


logging.basicConfig(filename='example.log')
logging.getLogger().setLevel(logging.INFO)

class Case1Bot(UTCBot):

    async def handle_round_started(self):
        '''
        This function is called when the round is started. You should do your setup here, and
        start any tasks that should be running for the rest of the round.
        '''
        self.rain = []

        self.fairs = {}
        self.order_book = {}
        self.pos = {}
        self.spread = {} # the spreads i will post
        self.state = {} # sell, normal, buy: inventory control state
        self.order_ids = {}
        self.time_step = 0
        self.pnl = 0
        self.history = {}
        for month in CONTRACTS:
            self.order_ids[month+' bid'] = ''
            self.order_ids[month+' ask'] = ''

            self.fairs[month] = 330

            self.order_book[month] = {
                'Best Bid':{'Price':0,'Quantity':0},
                'Best Ask':{'Price':0,'Quantity':0}}
            
            self.pos[month] = 0
            self.spread[month] = (0,0)
            self.state[month] = "normal"
            self.history[month] = [0.1] * 5
        
        self.set_year()
        asyncio.create_task(self.update_quotes())

    def set_year(self):
        pass

    def update_fairs(self):
        # implemented in subclass
        pass

    def get_spread(self):
        # implemented in subclass
        # return a dict giving the newest quotes
        # [bid, ask, bidsize, asksize] for each contract
        return {}

    async def update_quotes(self):
        # impmlemented in subclass
        while True:
            print(self.pos)
            print(self.state)
            self.time_step+=1
            self.update_fairs()
            spread_info = self.get_spread()

            for contract in CONTRACTS:
                spread = spread_info[contract]

                bid_response = await self.modify_order(
                    self.order_ids[contract+' bid'],
                    contract,
                    pb.OrderSpecType.LIMIT,
                    pb.OrderSpecSide.BID,
                    spread[2],spread[0])

                ask_response = await self.modify_order(
                    self.order_ids[contract+' ask'],
                    contract,
                    pb.OrderSpecType.LIMIT,
                    pb.OrderSpecSide.ASK,
                    spread[3],spread[1])

                assert bid_response.ok
                self.order_ids[contract+' bid'] = bid_response.order_id  
                    
                assert ask_response.ok
                self.order_ids[contract+' ask'] = ask_response.order_id 

                # for tracking transactions 
                self.history[contract].pop(0)
                self.history[contract].append(0)
            
                print(f'{round(spread[0],2)}@{round(spread[1],2)},{spread[2]},{spread[3]}')

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
            self.pnl = update.pnl_msg.realized_pnl
            print("M2M pnl:", update.pnl_msg.m2m_pnl)

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
                logging.info(f'BUY {fill_msg.asset} x {update.fill_msg.filled_qty} @ 195')
                self.pos[fill_msg.asset] += update.fill_msg.filled_qty
                self.history[fill_msg.asset][-1] = 1
                
            else:
                logging.info(f'SELL {fill_msg.asset} x {update.fill_msg.filled_qty} @ 310')
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
                print(msg)
                if "ROUND_ENDED" in msg:
                   print(self.pos)
                
                liquidate = 0
                for contract in CONTRACTS:
                    # get expiry price
                    year_idx = int(self.year) - 2016
                    price = (EXPIRY_PRICE[contract])[year_idx]
                    liquidate += self.pos[contract] * price
                
                print(self.pnl)
                print(liquidate)
                print(update.generic_msg.message)

