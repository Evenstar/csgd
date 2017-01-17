function csgd(opfunc, x, config, state)
   local config = config or {}
   local state = state or config
   local lr = config.learningRate or 0.01
   local lrd = config.learningRateDecay or 0

   local beta = config.beta or 0.9
   local epsilon = config.epsilon or 1e-8
   local wd = config.weightDecay or 0

   local fx, dfdx = opfunc(x)

   if wd ~= 0 then
      dfdx:add(wd, x)
   end


    --initialization

    state.t = state.t or 0

    state.g = state.g or x.new(dfdx:size()):zero()

    state.h = state.h or x.new(dfdx:size()):zero()

    state.xi = state.xi or x.new(dfdx:size()):zero()

    state.psi = state.psi or x.new(dfdx:size()):zero()

    state.phi = state.phi or x.new(dfdx:size()):zero()

    state.a = state.a or x.new(dfdx:size()):zero()

    state.Sigma = state.Sigma or x.new(dfdx:size()):zero()

    state.numer = state.numer or x.new(dfdx:size()):zero()

    state.stepsize = state.stepsize or torch.add(x.new(dfdx:size()):zero(),lr)


   state.t = state.t + 1

--    --learning rate decay at all steps
    local clr = lr / (1+state.t*lrd)

    state.g:mul(beta):add(1-beta,dfdx)

    state.h:mul(beta):addcmul(1-beta,dfdx,dfdx)

    state.xi:mul(beta):add(1-beta,x)

    state.psi:mul(beta):addcmul(1-beta,x,x)

    state.phi:mul(beta):addcmul(1-beta,x,dfdx)

    state.a:copy(state.psi):addcmul(-1,state.xi,state.xi)

    state.numer:copy(state.phi):addcmul(-1,state.g,state.xi)

    state.a:cdiv(state.numer+epsilon)

    state.Sigma:copy(state.h):addcmul(-1,state.g,state.g)


    local  temp_ = torch.cmul(state.g,state.g)
    local temp_numer = temp_:clone():addcmul(clr, state.a,temp_)

    local temp_denom = torch.cmul(state.a,state.Sigma)

    local stepsize_ = temp_numer:cdiv(temp_denom+epsilon):cmin(clr):cmul(state.a:sign():cmax(0))  - state.a:sign():cmin(0)

    state.stepsize:mul(beta):add(1-beta,stepsize_)

   x:addcmul(-1,state.stepsize, dfdx)
   return x, {fx}
end
