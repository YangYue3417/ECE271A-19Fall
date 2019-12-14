function v = dozigzag(u,zig_order)
    v = zeros(size(u));
    for i = 1:64
        v(:,zig_order(i)) = u(:,i);
    end
end
