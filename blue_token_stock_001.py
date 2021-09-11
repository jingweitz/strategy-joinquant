# 标  题：股指合理的蓝筹股策略（聚宽量化平台 https://www.joinquant.com）
# 微  信: junlin_tiger
# 公众号：曹经纬

from kuanke.wizard import *
from jqdata import *
import numpy as np
import pandas as pd
import talib
import datetime

## 初始化函数，设定要操作的股票、基准等等
def initialize(context):
    # 设定基准
    set_benchmark('000300.XSHG')
    # 设定滑点
    set_slippage(FixedSlippage(0))
    # True为开启动态复权模式，使用真实价格交易
    set_option('use_real_price', True)
    # 设定成交量比例
    set_option('order_volume_ratio', 1)
    # 股票类交易手续费是：买入时佣金万分之三，卖出时佣金万分之三加千分之一印花税, 每笔交易佣金最低扣5块钱
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.0003, close_commission=0.0003, min_commission=5), type='stock')
    # 个股最大持仓比重
    g.security_max_proportion = 1
    # 选股频率
    g.check_stocks_refresh_rate = 10
    # 买入频率
    g.buy_refresh_rate = 10
    # 卖出频率
    g.sell_refresh_rate = 10
    # 最大建仓数量
    g.max_hold_stocknum = 5

    # 选股频率计数器
    g.check_stocks_days = 0
    # 买卖交易频率计数器
    g.buy_trade_days=0
    g.sell_trade_days=0
    # 获取未卖出的股票
    g.open_sell_securities = []
    # 卖出股票的dict
    g.selled_security_list={}
    # 择时
    g.stock_selector = StockSelector()

    # 股票筛选初始化函数
    check_stocks_initialize()
    # 股票筛选排序初始化函数
    check_stocks_sort_initialize()
    # 出场初始化函数
    sell_initialize()
    # 入场初始化函数
    buy_initialize()
    # 风控初始化函数
    risk_management_initialize()

    # 关闭提示
    log.set_level('order', 'info')

    # 运行函数
    run_daily(sell_every_day,'10:00') #卖出未卖出成功的股票
    run_daily(risk_management, '10:00') #风险控制
    run_daily(check_stocks, '10:00') #选股
    run_daily(trade, '10:00') #交易
    run_daily(selled_security_list_count, 'after_close') #卖出股票日期计数


## 股票筛选初始化函数
def check_stocks_initialize():
    # 是否过滤停盘
    g.filter_paused = True
    # 是否过滤退市
    g.filter_delisted = True
    # 是否只有ST
    g.only_st = False
    # 是否过滤ST
    g.filter_st = True
    # 股票池
    g.security_universe_index = ["000300.XSHG","all_a_securities"]
    g.security_universe_user_securities = []
    # 行业列表
    g.industry_list = ["801010","801020","801030","801040","801050","801080","801110","801120","801130","801140","801150","801160","801170","801180","801200","801210","801230","801710","801720","801730","801740","801750","801760","801770","801780","801790","801880","801890"]
    # 概念列表
    g.concept_list = []

## 股票筛选排序初始化函数
def check_stocks_sort_initialize():
    # 总排序准则： desc-降序、asc-升序
    g.check_out_lists_ascending = 'desc'

## 出场初始化函数
def sell_initialize():
    # 设定是否卖出buy_lists中的股票
    g.sell_will_buy = True

    # 固定出仓的数量或者百分比
    g.sell_by_amount = None
    g.sell_by_percent = None

## 入场初始化函数
def buy_initialize():
    # 是否可重复买入
    g.filter_holded = False

    # 委托类型
    g.order_style_str = 'by_cap_mean'
    g.order_style_value = 100

## 风控初始化函数
def risk_management_initialize():
    # 策略风控信号
    g.risk_management_signal = True

    # 策略当日触发风控清仓信号
    g.daily_risk_management = True

    # 单只最大买入股数或金额
    g.max_buy_value = None
    g.max_buy_amount = None


## 卖出未卖出成功的股票
def sell_every_day(context):
    g.open_sell_securities = list(set(g.open_sell_securities))
    open_sell_securities = [s for s in context.portfolio.positions.keys() if s in g.open_sell_securities]
    if len(open_sell_securities)>0:
        for stock in open_sell_securities:
            order_target_value(stock, 0)
    g.open_sell_securities = [s for s in g.open_sell_securities if s in context.portfolio.positions.keys()]
    return

## 风控
def risk_management(context):
    ### _风控函数筛选-开始 ###
    ### _风控函数筛选-结束 ###
    return

## 股票筛选
def check_stocks(context):
    if g.check_stocks_days%g.check_stocks_refresh_rate != 0:
        # 计数器加一
        g.check_stocks_days += 1
        return
    # 股票池赋值
    g.check_out_lists = get_security_universe(context, g.security_universe_index, g.security_universe_user_securities)
    # 行业过滤
    g.check_out_lists = industry_filter(context, g.check_out_lists, g.industry_list)
    # 概念过滤
    g.check_out_lists = concept_filter(context, g.check_out_lists, g.concept_list)
    # 过滤ST股票
    g.check_out_lists = st_filter(context, g.check_out_lists)
    # 过滤退市股票
    g.check_out_lists = delisted_filter(context, g.check_out_lists)
    # 财务筛选
    g.check_out_lists = financial_statements_filter(context, g.check_out_lists)
    # 行情筛选
    g.check_out_lists = situation_filter(context, g.check_out_lists)
    # 技术指标筛选
    g.check_out_lists = technical_indicators_filter(context, g.check_out_lists)
    # 形态指标筛选函数
    g.check_out_lists = pattern_recognition_filter(context, g.check_out_lists)
    # 其他筛选函数
    #g.check_out_lists = other_func_filter(context, g.check_out_lists)

    # 排序
    input_dict = get_check_stocks_sort_input_dict()
    g.check_out_lists = check_stocks_sort(context,g.check_out_lists,input_dict,g.check_out_lists_ascending)
    
    g.check_out_lists = g.check_out_lists[:int(g.max_hold_stocknum*1.6)]
    g.check_out_lists = other_func_filter(context, g.check_out_lists)

    # 计数器归一
    g.check_stocks_days = 1
    return

## 交易函数
def trade(context):
   # 初始化买入列表
    buy_lists = []

    # 买入股票筛选
    if g.buy_trade_days%g.buy_refresh_rate == 0:
        # 获取 buy_lists 列表
        buy_lists = g.check_out_lists
        # 过滤ST股票
        buy_lists = st_filter(context, buy_lists)
        # 过滤停牌股票
        buy_lists = paused_filter(context, buy_lists)
        # 过滤退市股票
        buy_lists = delisted_filter(context, buy_lists)
        # 过滤涨停股票
        buy_lists = high_limit_filter(context, buy_lists)

        ### _入场函数筛选-开始 ###
        ### _入场函数筛选-结束 ###

    # 卖出操作
    if g.sell_trade_days%g.sell_refresh_rate != 0:
        # 计数器加一
        g.sell_trade_days += 1
    else:
        # 卖出股票
        sell(context, buy_lists)
        # 计数器归一
        g.sell_trade_days = 1


    # 买入操作
    if g.buy_trade_days%g.buy_refresh_rate != 0:
        # 计数器加一
        g.buy_trade_days += 1
    else:
        # 卖出股票
        buy(context, buy_lists)
        # 计数器归一
        g.buy_trade_days = 1

## 卖出股票日期计数
def selled_security_list_count(context):
    g.daily_risk_management = True
    if len(g.selled_security_list)>0:
        for stock in g.selled_security_list.keys():
            g.selled_security_list[stock] += 1

##################################  选股函数群 ##################################

## 财务指标筛选函数
def financial_statements_filter(context, security_list):
    ### _财务指标筛选函数-开始 ###
    #security_list = financial_data_filter_qujian(security_list, indicator.inc_net_profit_year_on_year, (-100,500))
    security_list = financial_data_filter_qujian(security_list, indicator.net_profit_margin, (15,135))
    security_list = financial_data_filter_qujian(security_list, valuation.pe_ratio, (30,100))
    #security_list = financial_data_filter_qujian(security_list, income.total_profit, (1e9,1.5e9))
    ### _财务指标筛选函数-结束 ###

    # 返回列表
    return security_list

## 行情筛选函数
def situation_filter(context, security_list):
    ### _行情筛选函数-开始 ###
    ### _行情筛选函数-结束 ###

    # 返回列表javascript:void(0)
    return security_list

## 技术指标筛选函数
def technical_indicators_filter(context, security_list):
    return security_list
    
    if check_for_baodie('000300.XSHG', 0, 2.5):
        return []
    
    
    periods = [15, 17, 20, 25, 30, 37, 45, 55, 70, 90]
    for period in periods:
        if RSI_judge_qujian('000300.XSHG', (53,99), period):
            return security_list
            
    return []

## 形态指标筛选函数
def pattern_recognition_filter(context, security_list):
    ### _形态指标筛选函数-开始 ###
    ### _形态指标筛选函数-结束 ###

    # 返回列表
    return security_list

## 其他方式筛选函数
def other_func_filter(context, security_list):
    ### _其他方式筛选函数-开始 ###
    security_list = [security for security in security_list if check_for_benchmark(context, security)]
    ### _其他方式筛选函数-结束 ###

    # 返回列表
    return security_list

# 获取选股排序的 input_dict
def get_check_stocks_sort_input_dict():
    input_dict = {
        #valuation.pe_ratio:('desc',0.5),
        valuation.market_cap:('desc',1),
        indicator.gross_profit_margin:('desc',1),
        income.net_profit:('desc',1),
        #finance.STK_EMPLOYEE_INFO.employee:('asc',1),
        #finance.STK_EMPLOYEE_INFO.retirement:('asc',1),
        }
    # 返回结果
    return input_dict

##################################  交易函数群 ##################################
# 交易函数 - 出场
def sell(context, buy_lists):
    # 获取 sell_lists 列表
    init_sl = context.portfolio.positions.keys()
    sell_lists = context.portfolio.positions.keys()

    # 判断是否卖出buy_lists中的股票
    if not g.sell_will_buy:
        sell_lists = [security for security in sell_lists if security not in buy_lists]

    ### _出场函数筛选-开始 ###
    ### _出场函数筛选-结束 ###

    # 卖出股票
    if len(sell_lists)>0:
        for stock in sell_lists:
            sell_by_amount_or_percent_or_none(context,stock, g.sell_by_amount, g.sell_by_percent, g.open_sell_securities)

    # 获取卖出的股票, 并加入到 g.selled_security_list中
    selled_security_list_dict(context,init_sl)

    return

# 交易函数 - 入场
def buy(context, buy_lists):
    # 风控信号判断
    if not g.risk_management_signal:
        return

    # 判断当日是否触发风控清仓止损
    if not g.daily_risk_management:
        return
    # 判断是否可重复买入
    buy_lists = holded_filter(context,buy_lists)

    # 获取最终的 buy_lists 列表
    Num = g.max_hold_stocknum - len(context.portfolio.positions)
    buy_lists = buy_lists[:Num]

    # 买入股票
    if len(buy_lists)>0:
        # 分配资金
        result = order_style(context,buy_lists,g.max_hold_stocknum, g.order_style_str, g.order_style_value)
        for stock in buy_lists:
            if len(context.portfolio.positions) < g.max_hold_stocknum:
                # 获取资金
                Cash = result[stock]
                # 判断个股最大持仓比重
                value = judge_security_max_proportion(context,stock,Cash,g.security_max_proportion)
                # 判断单只最大买入股数或金额
                amount = max_buy_value_or_amount(stock,value,g.max_buy_value,g.max_buy_amount)
                # 下单
                order(stock, amount, MarketOrderStyle())
    return

###################################  公用函数群 ##################################
## 排序
def check_stocks_sort(context,security_list,input_dict,ascending='desc'):
    if (len(security_list) == 0) or (len(input_dict) == 0):
        return security_list
    else:
        # 生成 key 的 list
        idk = list(input_dict.keys())
        # 生成矩阵
        a = pd.DataFrame()
        for i in idk:
            b = get_sort_dataframe(security_list, i, input_dict[i])
            a = pd.concat([a,b],axis = 1)
        # 生成 score 列
        a['score'] = a.sum(1,False)
        # 根据 score 排序
        if ascending == 'asc':# 升序
            if hasattr(a, 'sort'):
                a = a.sort(['score'],ascending = True)
            else:
                a = a.sort_values(['score'],ascending = True)
        elif ascending == 'desc':# 降序
            if hasattr(a, 'sort'):
                a = a.sort(['score'],ascending = False)
            else:
                a = a.sort_values(['score'],ascending = False)
        # 返回结果
        return list(a.index)

## 过滤同一标的继上次卖出N天不再买入
def filter_n_tradeday_not_buy(security, n=0):
    try:
        if (security in g.selled_security_list.keys()) and (g.selled_security_list[security]<n):
            return False
        return True
    except:
        return True

## 是否可重复买入
def holded_filter(context,security_list):
    if not g.filter_holded:
        security_list = [stock for stock in security_list if stock not in context.portfolio.positions.keys()]
    # 返回结果
    return security_list

## 卖出股票加入dict
def selled_security_list_dict(context,security_list):
    selled_sl = [s for s in security_list if s not in context.portfolio.positions.keys()]
    if len(selled_sl)>0:
        for stock in selled_sl:
            g.selled_security_list[stock] = 0

## 过滤停牌股票
def paused_filter(context, security_list):
    if g.filter_paused:
        current_data = get_current_data()
        security_list = [stock for stock in security_list if not current_data[stock].paused]
    # 返回结果
    return security_list

## 过滤退市股票
def delisted_filter(context, security_list):
    if g.filter_delisted:
        current_data = get_current_data()
        security_list = [stock for stock in security_list if not (('退' in current_data[stock].name) or ('*' in current_data[stock].name))]
    # 返回结果
    return security_list


## 过滤ST股票
def st_filter(context, security_list):
    if g.only_st:
        current_data = get_current_data()
        security_list = [stock for stock in security_list if current_data[stock].is_st]
    else:
        if g.filter_st:
            current_data = get_current_data()
            security_list = [stock for stock in security_list if not current_data[stock].is_st]
    # 返回结果
    return security_list

# 过滤涨停股票
def high_limit_filter(context, security_list):
    current_data = get_current_data()
    security_list = [stock for stock in security_list if not (current_data[stock].day_open >= current_data[stock].high_limit)]
    # 返回结果
    return security_list

# 获取股票股票池
def get_security_universe(context, security_universe_index, security_universe_user_securities):
    temp_index = []
    for s in security_universe_index:
        if s == 'all_a_securities':
            temp_index += list(get_all_securities(['stock'], context.current_dt.date()).index)
        else:
            temp_index += get_index_stocks(s)
    for x in security_universe_user_securities:
        temp_index += x
    return  sorted(list(set(temp_index)))

# 行业过滤
def industry_filter(context, security_list, industry_list):
    if len(industry_list) == 0:
        # 返回股票列表
        return security_list
    else:
        securities = []
        for s in industry_list:
            temp_securities = get_industry_stocks(s)
            securities += temp_securities
        security_list = [stock for stock in security_list if stock in securities]
        # 返回股票列表
        return security_list

# 概念过滤
def concept_filter(context, security_list, concept_list):
    if len(concept_list) == 0:
        return security_list
    else:
        securities = []
        for s in concept_list:
            temp_securities = get_concept_stocks(s)
            securities += temp_securities
        security_list = [stock for stock in security_list if stock in securities]
        # 返回股票列表
        return security_list

#自定义函数
def check_for_benchmark(context, security):
    #period = 30 + int((context.current_dt.month - 1) * 20.5) % 213
    #return RSI_judge_qujian(security, (50,99), 15) or RSI_judge_qujian(security, (50,99), period)
    
    #return RSI_judge_qujian(security, (50,99), 15) or RSI_judge_qujian(security, (50,99), 30) or RSI_judge_qujian(security, (50,99), 60) or RSI_judge_qujian(security, (50,99), 90)
    
    return g.stock_selector.check_for_could_trade(context, security, 5, 0, 3, 10)
    #return True
    
def get_last_price(context, symbol):
    if get_current_data()[symbol].paused:
        #log.info('%s 今日停牌' % symbol)
        return -1.0
    
    begintime = context.current_dt
    endtime = begintime
    hst = attribute_history(symbol, 2, unit='1m', fields=['close'], fq='post')
    close_list = hst['close'].tolist()
    if close_list is not None and len(close_list) >= 1:
        return close_list[-1]
    else:
        return -1.0    
    
    
class StockSelector(object):
    def __init__(self):
        self.risk_symbol_dict = {}
        self.symbol_finance_dict = {}
        self.symbol_trade_status = {}
        

    def check_for_could_trade(self, context, symbol, period, begin_index, min_atr_rate, risk_days):
        """
        attribute_history 函数文档
        attribute_history(security, count, unit='1d', fields=['open', 'close', 'high', 'low', 'volume', 'money'], skip_paused=True, df=True, fq='pre')
        """

        # attribute_history只能获取到前一天的收盘价
        hst = attribute_history(symbol, period, unit='1d', fields=['open', 'high', 'low', 'close'], fq='post')
        close_list = hst['close'].tolist()
        
        if len(close_list) == period:
            
            # 获取当前价格，并添加到close_list中
            close = get_last_price(context, symbol)
            close_list.append(close)
            
        #atr_dict = ATR(symbol, 60)
        #atr = atr_dict[symbol].tolist()[-1]    
        
        period = 61
        hst = attribute_history(symbol, period, unit='1d', fields=['open', 'high', 'low', 'close'], fq='post')

        
        #last_price = get_last_price(context, symbol)
  
        close = np.array(hst['close'])
        high = np.array(hst['high'])
        low = np.array(hst['low'])
        
        if close[-1] is None or math.isnan(close[-1]):
            return False
        
        #close = np.insert(close, -1, last_price, axis=0)
        #high = np.insert(high, -1, last_price, axis=0)
        #low = np.insert(low, -1, last_price, axis=0)
        
        # 获取最新的ATR值
        atr_list = talib.ATR(high, low, close, timeperiod=60)
        #print('atr_list:', close)

        atr = atr_list[-1]

        min_profit_rate = -atr * min_atr_rate / close_list[-1]
        min_profit_rate = max(-0.08, min(-0.02, min_profit_rate))
        
            
            
        is_baodie = False    
        min_profit_rate2 = 255
        #for i in range(len(close_list)):
        for i in range(begin_index, len(close_list), 1):    
            size = len(close_list) - i
            profit_rate = (close_list[-1] - close_list[i]) / (close_list[i]) / (i + 1)
            min_profit_rate2 = min(min_profit_rate2, profit_rate)
            if profit_rate < min_profit_rate and close_list[-1] > 0:
            #if (profit_rate < min_profit_rate or profit_rate > max_profit_rate) and close_list[-1] > 0:
                log.info('%s %s 暴跌[%d %.2f%%]，暂时被踢出股票池' % (symbol, get_security_info(symbol).display_name, i + 1, profit_rate * 100))
                is_baodie = True
                break
            
        print('min_profit_rate[%.2f %.2f]' % (min_profit_rate * 100, min_profit_rate2 * 100))

        
        #+----------+----------+----------+----------+----------+----------+----------+----------+
        # 暴跌后N天内不交易
        if is_baodie:
            self.risk_symbol_dict[symbol] = risk_days
        elif self.risk_symbol_dict.get(symbol) is not None:
            if self.risk_symbol_dict[symbol] < 0:
                del self.risk_symbol_dict[symbol]
        
            
        could_trade = True
        if self.risk_symbol_dict.get(symbol) is not None:
            if self.risk_symbol_dict[symbol] > 0:
                log.info('%s %s 暴跌风险期[%d]，暂时被踢出股票池' % (symbol, get_security_info(symbol).display_name, self.risk_symbol_dict[symbol]))
                could_trade = False
                
            self.risk_symbol_dict[symbol] = self.risk_symbol_dict[symbol] - 1
        

        if self.symbol_trade_status.get(symbol) is None:
            self.symbol_trade_status[symbol] = 'close'

        '''
        period = 30 + (context.current_dt.day + int((context.current_dt.month -1) * 20.5)) % 213
        if self.symbol_trade_status[symbol] == 'close':
            could_trade_rsi = False
            if RSI_judge_qujian(symbol, (45,99), period) and RSI_judge_qujian(symbol, (52,99), 15):
                self.symbol_trade_status[symbol] = 'open'
                could_trade_rsi = True
        elif self.symbol_trade_status[symbol] == 'open':
            could_trade_rsi = True
            if RSI_judge_qujian(symbol, (0,42), period):
                self.symbol_trade_status[symbol] = 'close'
                could_trade_rsi = False
        '''
        
        period = 30 + (context.current_dt.day + int((context.current_dt.month -1) * 20.5)) % 213
        if self.symbol_trade_status[symbol] == 'close':
            could_trade_rsi = False
            if check_for_rsi(symbol, period, 45, 99) and check_for_rsi(symbol, 15, 52, 99):
                self.symbol_trade_status[symbol] = 'open'
                could_trade_rsi = True
        elif self.symbol_trade_status[symbol] == 'open':
            could_trade_rsi = True
            if check_for_rsi(symbol, period, 0, 42):
                self.symbol_trade_status[symbol] = 'close'
                could_trade_rsi = False
            
        
        
        could_trade = could_trade and could_trade_rsi
        #if not could_trade and self.risk_symbol_dict[symbol] < risk_days - 5:
        #    could_trade = RSI_judge_qujian(symbol, (50,99), 60)
        
            
        return could_trade     
    
def check_for_rsi(symbol, period, rsi_min, rsi_max, show_rsi=False):
    hst = attribute_history(symbol, period + 1, '1d', ['close'])
    close = [float(x) for x in hst['close']]
    if (math.isnan(close[0]) or math.isnan(close[-1])):
        return False
    
    rsi = talib.RSI(np.array(close), timeperiod=period)[-1]  
    if (show_rsi):
        record(RSI=max(0,(rsi-50)))  

    return (rsi_min < rsi < rsi_max)
    
    
def check_for_baodie(symbol, begin_index, min_atr_rate):
    period = 61
    hst = attribute_history(symbol, period, unit='1d', fields=['open', 'high', 'low', 'close'], fq='post')
    
    close_list = hst['close'].tolist()
    
    #last_price = get_last_price(context, symbol)

    close = np.array(hst['close'])
    high = np.array(hst['high'])
    low = np.array(hst['low'])
    
    if close[-1] is None or math.isnan(close[-1]):
        return False
    
    #close = np.insert(close, -1, last_price, axis=0)
    #high = np.insert(high, -1, last_price, axis=0)
    #low = np.insert(low, -1, last_price, axis=0)
    
    # 获取最新的ATR值
    atr_list = talib.ATR(high, low, close, timeperiod=60)
    #print('atr_list:', close)

    atr = atr_list[-1]

    min_profit_rate = -atr * min_atr_rate / close_list[-1]
    min_profit_rate = max(-0.08, min(-0.03, min_profit_rate))
    
        
        
    is_baodie = False    
    min_profit_rate2 = 255
    #for i in range(len(close_list)):
    for i in range(begin_index, len(close_list), 1):    
        size = len(close_list) - i
        profit_rate = (close_list[-1] - close_list[i]) / (close_list[i]) / (i + 1)
        min_profit_rate2 = min(min_profit_rate2, profit_rate)
        if profit_rate < min_profit_rate and close_list[-1] > 0:
        #if (profit_rate < min_profit_rate or profit_rate > max_profit_rate) and close_list[-1] > 0:
            log.info('%s %s 暴跌[%d %.2f%%]' % (symbol, get_security_info(symbol).display_name, i + 1, profit_rate * 100))
            is_baodie = True
            break
        
    return is_baodie
    
