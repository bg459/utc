
# python sample_bot.py --year $1 & ./xchange -pricepath $1 -debug case1
python mm_bot_v1.py $1 & ./xchange -pricepath=$1 case1