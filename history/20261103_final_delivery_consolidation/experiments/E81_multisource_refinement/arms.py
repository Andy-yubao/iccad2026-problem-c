"""One preregistered small-P / larger-F combination, not a Q checkpoint search."""
ARMS={'topology':{'width':96,'layers':12,'rectangle':True,'ffn_dropout':.05,
                 'corrector_width':192,'message_width':64,'max_correction':.15,'alternative_width':192}}
def model_config(arm):return dict(ARMS[arm])
