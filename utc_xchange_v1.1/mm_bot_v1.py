from utc_bot import UTCBot, start_bot
import proto.utc_bot as pb
import betterproto
import sys
import asyncio
from case1_bot import Case1Bot

CONTRACTS = ["LBSJ","LBSM", "LBSQ", "LBSV", "LBSZ"]
ORDER_SIZE = 20
SPREAD = 2

EXPIRY_PRICE = { #2016-2021
    "LBSJ": [221.23,210.73,185.91,220.15,219.58,213.2],
    "LBSM": [207.55,209.6,280.91,281.31,255.24,208.49],
    "LBSQ": [246.63,276.45,260.07,227.84,242.59,258.71],
    "LBSV": [262.2,296.24,244.98,228.33,242.1,270.32],
    "LBSZ": [282.58,260.49,265.63,280.89,311.71,330.02]
}


class MMBoT(Case1Bot):

    def set_year(self):
        self.year = year # for pnl
    def update_fairs(self):
        # updates fairs, also sets the state to offset positions
        for month in CONTRACTS:
            self.fairs[month] = 300



    def get_spread(self):
        
        our_spread = {}
        for month in CONTRACTS:
            fair = self.fairs[month]
            
            (lo, hi) = self.spread[month]
            if self.pos[month] >= 5:
                if self.state[month] != "sell":
                    hi = lo + (hi-lo)*(1/4) # moving to sell state
                else:
                    hi -= 1
                bid_size, ask_size = 1, self.pos[month]+1
                self.state[month] = "sell" # want to sell off
            
            elif self.pos[month] <= -5:
                if self.state[month] != "buy":
                    lo = hi - (hi-lo) * (3/4)
                else:
                    lo += 1
                bid_size, ask_size = abs(self.pos[month])+1, 1
                self.state[month] = "buy"
            else:
                self.state[month] = "normal"
                lo, hi = fair - 40, fair + 40

                bid_size, ask_size = 10, 10
            
            self.spread[month] = (lo, hi)
        
            our_spread[month] = [lo, hi, bid_size, ask_size]

        return our_spread


if __name__ == "__main__":
    year = sys.argv[1]
    print(year)
    start_bot(MMBoT)