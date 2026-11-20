"""Single fixed FM objective, shared by optimization, monitoring and selection."""
ARMS=('fm',)
def configuration(arm='fm'):
    if arm!='fm':raise ValueError(arm)
    return dict(objective='linear conditional flow matching',
        formula='mean_case(mean_valid_module(mean_xyz((v_theta-(z1-z0))**2)))',
        state='train-only frozen equal-case axis-symmetric affine normalization',
        noise='N(0,I)',time='uniform0..1',
        monitor='four fixed member-keyed draws with stratified times;equal average',
        task_loss=False,fixed_preplaced_channels='all predicted;no overwrite',test_read=False)
