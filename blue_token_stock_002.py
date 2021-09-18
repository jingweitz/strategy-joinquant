# 标  题：蓝筹股策略系列（2）（聚宽量化平台 https://www.joinquant.com）
# 微  信: junlin_tiger
# 公众号：曹经纬


from jqdata import *
from kuanke.wizard import *

import six
from numpy import mean
import numpy as np
import pandas as pd

from abc import ABCMeta, abstractmethod


def initialize(context):
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    log.info('策略启动')


    # 股票类每笔交易时的手续费是：买入时佣金万分之三，卖出时佣金万分之三加千分之一印花税, 每笔交易佣金最低扣5块钱
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.0003, close_commission=0.0003, min_commission=5), type='stock')
    
    g.strategy_manager = StrategyManager()

    ## 运行函数（reference_security为运行时间的参考标的；传入的标的只做种类区分，因此传入'000300.XSHG'或'510300.XSHG'是一样的）
    run_daily(timer_event, time='13:30', reference_security='000300.XSHG')

    
    
def timer_event(context):
    g.strategy_manager.timer_event(context)


#--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
class Configure(object):
    def __init__(self):
        self.select_symbol_peroid = 30
        self.use_remote_algorithm = False


class StrategyBase(object):
    def __init__(self, root, name, config):
        self.root = root
        self.name = name
        self.config = config
        self.trade_days = 0
        self.symbol_list = []
        #print('StrategyBase.__init__')
        pass
    
    # 风控管理
    @abstractmethod
    def risk_control(self, context):
        return True    
    
    
    # 选股 
    @abstractmethod
    def select_symbol(self, context):
        return []
    
    # 定时器    
    def timer_event(self, context):
        self.risk_control(context)
        self.select_symbol(context)
        
        
        
#--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
class Strategy001(StrategyBase):
    def __init__(self, root, name, config):
        StrategyBase.__init__(self, root, name, config)
        pass
    
    # 风控管理
    def risk_control(self, context):
        #print('Strategy001.risk_control %r %s' % (context.current_dt.strftime('%Y-%m-%d %H:%M:%S'), self.root.name))
        
        # TODO ...
        return True    
    
    
    def select_symbol(self, context):
        self.trade_days += 1
        if self.trade_days % self.config.select_symbol_period != 0 and self.trade_days > 1:
            return self.symbol_list
        
        # 原始股票池
        symbol_list = get_security_universe(context, ['000300.XSHG'], []) # 沪深300   
        #symbol_list = get_security_universe(context, ['000905.XSHG'], []) # 中证500
        
        q = query(valuation.code, valuation.pb_ratio, valuation.market_cap, indicator.roe, indicator.inc_total_revenue_year_on_year, indicator.inc_operation_profit_year_on_year, indicator.inc_net_profit_year_on_year
            ).filter(
                valuation.code.in_(symbol_list))
                
        df =  get_fundamentals(q)   
        
        
        if df is None or df.shape[0] == 0:
            log.error('--------------------------------------------')
            return self.symbol_list
            
        df.rename(columns={
            'inc_total_revenue_year_on_year': 'ystb',
            'inc_operation_profit_year_on_year': 'lrtb',
            'inc_net_profit_year_on_year': 'jlrtb',
            'anon_1': 'fuzhailv'}, inplace=True)      
            
        #删除空值
        df = df.dropna(axis=0, how='all')   
        
        if self.config.use_remote_algorithm:
            # TODO ...
            pass
        else:
            df = df[(df.roe > 4) & (df.pb_ratio > 0) & (df.pb_ratio < 15) & (df.ystb > 10) & (df.ystb < 1000)].sort_values('roe', ascending=False)
            df = df[:max(50, min(100, self.root.max_stock_count * 10))]
            
            
            if df is None or df.shape[0] == 0:
                return self.symbol_list
            
            chg_list = []
            for index, row in df.iterrows():
                chg_list.append(get_n_day_chg(row.code, 30))
                
            df['chg'] = chg_list
            df.index = df['code'].values

            df['score'] = df[['market_cap', 'chg', 'ystb']].rank().T.apply(lambda x : x[0] * 2 + x[1] * 2 + x[2] * 1)

            
            #按得分进行排序，取指定数量的股票
            df = df.sort_values('score', ascending=False)
            self.symbol_list = df.index
        
        self.symbol_list = [symbol for symbol in self.symbol_list if n_day_chg_dayu(symbol, 90, 0.1)]
        
        self.symbol_list = self.symbol_list[:int(self.root.max_stock_count / 2)] 
        
        log.info('最新信号 ==> %s' % self.name)
        for symbol in self.symbol_list:
            log.info('%s  %s' % (symbol, get_security_info(symbol).display_name))  
            
        return self.symbol_list
        
      
class Strategy002(StrategyBase):
    def __init__(self, root, name, config):
        StrategyBase.__init__(self, root, name, config)
        pass
    
    # 风控管理
    def risk_control(self, context):
        #print('Strategy002.risk_control %r %s' % (context.current_dt.strftime('%Y-%m-%d %H:%M:%S'), self.root.name))
        
        # TODO ...
        return True    
    
    
    def select_symbol(self, context):
        self.trade_days += 1
        if self.trade_days % self.config.select_symbol_period != 0 and self.trade_days > 1:
            return self.symbol_list
        
        # 原始股票池
        symbol_list = get_security_universe(context, ['000300.XSHG'], []) # 沪深300   
        #symbol_list = get_security_universe(context, ['000905.XSHG'], []) # 中证500
        
        q = query(valuation.code, valuation.pb_ratio, valuation.market_cap, indicator.roe, indicator.inc_total_revenue_year_on_year, indicator.inc_operation_profit_year_on_year, indicator.inc_net_profit_year_on_year
            ).filter(
                valuation.code.in_(symbol_list))
                
        df =  get_fundamentals(q)   
        
        
        if df is None or df.shape[0] == 0:
            log.error('--------------------------------------------')
            return self.symbol_list
            
        df.rename(columns={
            'inc_total_revenue_year_on_year': 'ystb',
            'inc_operation_profit_year_on_year': 'lrtb',
            'inc_net_profit_year_on_year': 'jlrtb',
            'anon_1': 'fuzhailv'}, inplace=True)      
            
        #删除空值
        df = df.dropna(axis=0, how='all')   
        
        if self.config.use_remote_algorithm:
            # TODO ...
            pass
        else:
            df = df[(df.roe > 4) & (df.pb_ratio > 0) & (df.pb_ratio < 15) & (df.ystb > 10) & (df.ystb < 1000)].sort_values('roe', ascending=False)
            df = df[:max(50, min(100, self.root.max_stock_count * 10))]
            
            
            if df is None or df.shape[0] == 0:
                return self.symbol_list
            
            chg_list = []
            for index, row in df.iterrows():
                chg_list.append(get_n_day_chg(row.code, 30))
                
            df['chg'] = chg_list
            df.index = df['code'].values

            df['score'] = df[['market_cap', 'chg', 'ystb']].rank().T.apply(lambda x : x[0] * 2 + x[1] * 2 + x[2] * 1)

            
            #按得分进行排序，取指定数量的股票
            df = df.sort_values('score', ascending=False)
            self.symbol_list = df.index
        
        self.symbol_list = [symbol for symbol in self.symbol_list if n_day_chg_dayu(symbol, 90, -0.1)]
        
        self.symbol_list = self.symbol_list[:int(self.root.max_stock_count / 2)] 
        
        log.info('最新信号 ==> %s' % self.name)
        for symbol in self.symbol_list:
            log.info('%s  %s' % (symbol, get_security_info(symbol).display_name))  
            
        return self.symbol_list


#--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
class StrategyManager():
    def __init__(self):
        self.name = 'root'
        self.strategy_list = []
        self.max_stock_count = 10
        
        config = Configure()
        config.select_symbol_period = 10
        
        self.strategy_list.append(Strategy001(self, '沪深300指数增强', config))
        self.strategy_list.append(Strategy002(self, '沪深300指数增强', config))
        
        
    # 定时器    
    def timer_event(self, context):
        print('\n==========================================================\n')
        
        symbol_list = []
        for strategy in self.strategy_list:
            strategy.timer_event(context)
            symbol_list.extend(strategy.symbol_list)
            
        # TODO tet    
        #log.info('大盘预测[%s]' % '上涨' if rsi_judge('000300.XSHG', '1d', 15, (50, 100)) else '下跌')
        #log.info('held symbol：%s' % symbol_list)
        self.adjust_position(context, symbol_list, self.max_stock_count)
        
    
    @staticmethod
    def adjust_position_v1(context, pool, max_stock_count):
        cash = context.portfolio.total_value / max_stock_count #* StrategyManager.get_dynamic_turnover()
        hold_stock = context.portfolio.positions.keys()
        
        #卖出不在持仓中的股票
        for stock in hold_stock:
            if stock not in pool:
                order_target_value(stock, 0)
                
        #买入股票
        held_stock = context.portfolio.positions.keys() 
        for stock in pool:
            if len(held_stock) < max_stock_count + 1:
                order_target_value(stock, cash)
                
                
    @staticmethod
    def adjust_position(context, pool, max_stock_count):
        cash = context.portfolio.total_value / max_stock_count #* StrategyManager.get_dynamic_turnover()
        hold_stock = context.portfolio.positions.keys()
        
        #卖出不在持仓中的股票
        for stock in hold_stock:
            if stock not in pool:
                order_target_value(stock, 0)
                
        #买入股票
        #held_stock = context.portfolio.positions.keys() 
        pool_unique = set(pool)
        for stock in pool_unique:
            cash2 = cash * pool.count(stock)
            order_target_value(stock, cash2)                

            
    @staticmethod        
    def get_dynamic_turnover():
        count = 0
        periods = [15, 30, 45, 60, 90]
        for period in periods:
            if RSI_judge_qujian('000300.XSHG', (53, 99), period):
                count += 1
                
            if RSI_judge_qujian('000001.XSHG', (53, 99), period):    
                count += 1
        
        return count / len(periods) / 2            

            
#--------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
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

