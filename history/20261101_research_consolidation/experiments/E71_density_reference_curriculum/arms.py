"""One combined seed0 candidate; explanatory ablations deferred until improvement."""
ARMS={name:{'width':96,'layers':12,'rectangle':True,'ffn_dropout':.05,
            'port_mode':mode,'port_width':24,'port_scale':.1}
      for name,mode in (('full_ports','full'),)}


def model_config(arm):return dict(ARMS[arm])
